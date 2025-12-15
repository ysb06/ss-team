import asyncio
import logging
import time

from .attack import main
from .dataset import load_awesome_chatgpt_prompts

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    peek_count = 3
    max_prompt_count = 20  # 최대 프롬프트 수 제한 (원하는 값으로 조정 가능)

    # 전체 실험 시작 시간
    experiment_start_time = time.time()

    victim_prompts, known_prefixes = load_awesome_chatgpt_prompts("data/prompts.csv")

    # 결과 집계를 위한 변수
    total_prompts = len(victim_prompts)
    successful_prompts = 0  # peek_count만큼 모두 성공한 프롬프트 수
    total_correct_tokens = 0  # 전체 정확한 토큰 수
    total_attempted_tokens = total_prompts * peek_count  # 전체 시도한 토큰 수

    # 시간 측정을 위한 변수
    all_peek_one_token_times = []  # 모든 peek_one_token 호출 시간
    all_attack_request_times = []  # 모든 _send_attack_requests 호출 시간
    prompt_times = []  # 각 victim 프롬프트당 걸린 시간

    logger.info(f"[MAIN] Starting attacks on {total_prompts} prompts...")
    logger.info(f"[MAIN] Each prompt will attempt to extract {peek_count} tokens.")

    prompts_tested = 0  # 실제로 테스트한 프롬프트 수
    prompts_skipped = 0  # 에러로 인해 스킵된 프롬프트 수

    # 공격 실행
    try:
        for i, (victim_prompt, known_prefix) in enumerate(
            zip(victim_prompts, known_prefixes)
        ):
            if i >= max_prompt_count:
                logger.info(
                    f"[MAIN] Reached max prompt count of {max_prompt_count}. Stopping further attacks."
                )
                break
            
            print(f"\n[MAIN] Starting attack on prompt: {i+1}/{len(victim_prompts)}")

            # 각 프롬프트 처리를 개별적으로 try-except로 감싸기
            try:
                # 각 프롬프트 공격 시작 시간
                prompt_start_time = time.time()

                correct_count, extracted_tokens, peek_times, attack_times = main(
                    victim_prompt, known_prefix, peek_count=peek_count
                )

                # 각 프롬프트 공격 종료 시간
                prompt_elapsed = time.time() - prompt_start_time
                prompt_times.append(prompt_elapsed)

                # 시간 데이터 수집
                all_peek_one_token_times.extend(peek_times)
                all_attack_request_times.extend(attack_times)

                # 집계
                prompts_tested += 1
                total_correct_tokens += correct_count
                if correct_count == peek_count:
                    successful_prompts += 1
            except Exception as e:
                logger.error(f"[MAIN] Error occurred on prompt {i+1}: {e}")
                logger.info(f"[MAIN] Skipping prompt {i+1} and continuing with next prompt...")
                prompts_skipped += 1
                logger.info("[MAIN] Waiting for 2 minutes before continuing...")
                time.sleep(120)  # 잠시 대기 후 계속 진행
                continue
    except KeyboardInterrupt:
        print(
            "\n\n[MAIN] ⚠️  KeyboardInterrupt detected! Stopping and aggregating results so far..."
        )
        logger.warning("Attack interrupted by user. Generating partial results.")
    # 공격 끝

    # 전체 실험 종료 시간
    experiment_elapsed = time.time() - experiment_start_time

    # 최종 결과 출력 (정상 종료 또는 중단 모두)
    actual_attempted_tokens = prompts_tested * peek_count

    # 시간 통계 계산
    avg_peek_one_token_time = (
        sum(all_peek_one_token_times) / len(all_peek_one_token_times)
        if all_peek_one_token_times
        else 0
    )
    avg_attack_request_time = (
        sum(all_attack_request_times) / len(all_attack_request_times)
        if all_attack_request_times
        else 0
    )
    avg_prompt_time = sum(prompt_times) / len(prompt_times) if prompt_times else 0

    print("\n" + "=" * 80)
    print("FINAL RESULTS - Dataset Aggregation")
    if prompts_tested < total_prompts:
        print(f"(Interrupted - Partial Results)")
    print("=" * 80)
    print(f"\nDataset size:              {total_prompts}")
    print(f"Prompts tested:            {prompts_tested}/{total_prompts}")
    print(f"Prompts skipped (errors):  {prompts_skipped}")
    print(f"Tokens per prompt:         {peek_count}")
    print(f"Total attempted tokens:    {actual_attempted_tokens}")
    print(f"\n--- Prompt Success Rate ---")
    print(f"Fully successful prompts:  {successful_prompts}/{prompts_tested}")
    if prompts_tested > 0:
        print(
            f"Prompt success rate:       {successful_prompts/prompts_tested*100:.2f}%"
        )
    else:
        print(f"Prompt success rate:       N/A (no prompts tested)")
    print(f"\n--- Token Success Rate ---")
    print(
        f"Correct tokens extracted:  {total_correct_tokens}/{actual_attempted_tokens}"
    )
    if actual_attempted_tokens > 0:
        print(
            f"Token success rate:        {total_correct_tokens/actual_attempted_tokens*100:.2f}%"
        )
    else:
        print(f"Token success rate:        N/A (no tokens attempted)")

    print(f"\n--- Timing Statistics ---")
    print(
        f"Total experiment time:          {experiment_elapsed:.2f} seconds ({experiment_elapsed/60:.2f} minutes)"
    )
    print(f"Avg. time per prompt:           {avg_prompt_time:.2f} seconds")
    print(f"Avg. peek_one_token time:       {avg_peek_one_token_time:.2f} seconds")
    print(f"Avg. _send_attack_requests time: {avg_attack_request_time:.2f} seconds")

    if prompt_times:
        print(f"\n--- Prompt Time Details ---")
        print(f"Min prompt time:                {min(prompt_times):.2f} seconds")
        print(f"Max prompt time:                {max(prompt_times):.2f} seconds")

    print("=" * 80 + "\n")
    logger.info("[MAIN] Attack process completed.")
