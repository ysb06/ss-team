import logging
import time

from .attack import main_single, main_double
from .dataset import load_awesome_chatgpt_prompts

logger = logging.getLogger(__name__)

def double_victim_experiment():
    """
    2 Victim - 2 Attacker simultaneous attack experiment
    Attack prompts in pairs of 2 and aggregate results
    """
    peek_count = 3
    max_prompt_pairs = 20  # Maximum prompt pair limit

    # Overall experiment start time
    experiment_start_time = time.time()

    victim_prompts, known_prefixes = load_awesome_chatgpt_prompts("data/prompts.csv")

    # Variables for result aggregation
    total_prompts = len(victim_prompts)
    total_pairs = total_prompts // 2  # Number of pairs
    
    # Variables for single attack comparison
    total_correct_tokens_v1 = 0  # Victim 1 correct token count
    total_correct_tokens_v2 = 0  # Victim 2 correct token count
    total_attempted_tokens_v1 = 0  # Victim 1 attempted token count
    total_attempted_tokens_v2 = 0  # Victim 2 attempted token count
    
    successful_prompts_v1 = 0  # Number of prompts fully successful for Victim 1
    successful_prompts_v2 = 0  # Number of prompts fully successful for Victim 2

    # Variables for time measurement
    all_peek_one_token_times = []
    all_attack_request_times = []
    pair_times = []  # Time taken for each pair processing

    logger.info(f"[MAIN] Starting DOUBLE attacks on {total_prompts} prompts ({total_pairs} pairs)...")
    logger.info(f"[MAIN] Each prompt will attempt to extract {peek_count} tokens.")

    pairs_tested = 0  # Number of pairs actually tested
    pairs_skipped = 0  # Number of pairs skipped due to errors

    # Execute attacks
    try:
        for pair_idx in range(0, min(total_pairs * 2, len(victim_prompts)), 2):
            if pairs_tested >= max_prompt_pairs:
                logger.info(
                    f"[MAIN] Reached max prompt pairs of {max_prompt_pairs}. Stopping further attacks."
                )
                break

            # Stop if pair is not complete
            if pair_idx + 1 >= len(victim_prompts):
                logger.info(f"[MAIN] Odd number of prompts. Skipping last prompt.")
                break

            victim_prompt1 = victim_prompts[pair_idx]
            known_prefix1 = known_prefixes[pair_idx]
            victim_prompt2 = victim_prompts[pair_idx + 1]
            known_prefix2 = known_prefixes[pair_idx + 1]

            print(f"\n[MAIN] Starting DOUBLE attack on prompt pair: {pairs_tested+1}/{min(max_prompt_pairs, total_pairs)}")
            print(f"[MAIN]   Victim 1: Prompt {pair_idx+1}")
            print(f"[MAIN]   Victim 2: Prompt {pair_idx+2}")

            # Wrap each pair processing in individual try-except
            try:
                # Start time for each pair attack
                pair_start_time = time.time()

                result1, result2 = main_double(
                    victim_prompt1, known_prefix1,
                    victim_prompt2, known_prefix2,
                    peek_count=peek_count,
                    waiting_time=0  # Remove wait time during experiment
                )

                # End time for each pair attack
                pair_elapsed = time.time() - pair_start_time
                pair_times.append(pair_elapsed)

                # Unpack results
                correct_count1, extracted_tokens1, peek_times1, attack_times1 = result1
                correct_count2, extracted_tokens2, peek_times2, attack_times2 = result2

                # Collect time data
                all_peek_one_token_times.extend(peek_times1)
                all_peek_one_token_times.extend(peek_times2)
                all_attack_request_times.extend(attack_times1)
                all_attack_request_times.extend(attack_times2)

                # Aggregate results
                pairs_tested += 1
                
                # Victim 1 aggregation
                total_attempted_tokens_v1 += peek_count
                total_correct_tokens_v1 += correct_count1
                if correct_count1 == peek_count:
                    successful_prompts_v1 += 1
                
                # Victim 2 aggregation
                total_attempted_tokens_v2 += peek_count
                total_correct_tokens_v2 += correct_count2
                if correct_count2 == peek_count:
                    successful_prompts_v2 += 1

            except Exception as e:
                logger.error(f"[MAIN] Error occurred on prompt pair {pairs_tested+1}: {e}")
                logger.info(
                    f"[MAIN] Skipping prompt pair {pairs_tested+1} and continuing with next pair..."
                )
                pairs_skipped += 1
                logger.info("[MAIN] Waiting for 2 minutes before continuing...")
                time.sleep(120)  # Wait briefly before continuing
                continue
    except KeyboardInterrupt:
        print(
            "\n\n[MAIN] ⚠️  KeyboardInterrupt detected! Stopping and aggregating results so far..."
        )
        logger.warning("Attack interrupted by user. Generating partial results.")
    # Attack finished

    # Overall experiment end time
    experiment_elapsed = time.time() - experiment_start_time

    # Print final results (both normal termination and interruption)
    total_prompts_tested = pairs_tested * 2
    total_attempted_tokens = total_attempted_tokens_v1 + total_attempted_tokens_v2
    total_correct_tokens = total_correct_tokens_v1 + total_correct_tokens_v2

    # Calculate time statistics
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
    avg_pair_time = sum(pair_times) / len(pair_times) if pair_times else 0

    print("\n" + "=" * 80)
    print("FINAL RESULTS - DOUBLE VICTIM Dataset Aggregation")
    if pairs_tested < total_pairs:
        print(f"(Interrupted - Partial Results)")
    print("=" * 80)
    print(f"\nDataset size:              {total_prompts} prompts ({total_pairs} pairs)")
    print(f"Prompt pairs tested:       {pairs_tested}/{min(max_prompt_pairs, total_pairs)}")
    print(f"Prompt pairs skipped:      {pairs_skipped}")
    print(f"Total prompts tested:      {total_prompts_tested}")
    print(f"Tokens per prompt:         {peek_count}")
    print(f"Total attempted tokens:    {total_attempted_tokens}")
    
    print(f"\n--- Victim 1 (First in each pair) ---")
    print(f"Fully successful prompts:  {successful_prompts_v1}/{pairs_tested}")
    if pairs_tested > 0:
        print(f"Prompt success rate:       {successful_prompts_v1/pairs_tested*100:.2f}%")
    print(f"Correct tokens:            {total_correct_tokens_v1}/{total_attempted_tokens_v1}")
    if total_attempted_tokens_v1 > 0:
        print(f"Token success rate:        {total_correct_tokens_v1/total_attempted_tokens_v1*100:.2f}%")
    
    print(f"\n--- Victim 2 (Second in each pair) ---")
    print(f"Fully successful prompts:  {successful_prompts_v2}/{pairs_tested}")
    if pairs_tested > 0:
        print(f"Prompt success rate:       {successful_prompts_v2/pairs_tested*100:.2f}%")
    print(f"Correct tokens:            {total_correct_tokens_v2}/{total_attempted_tokens_v2}")
    if total_attempted_tokens_v2 > 0:
        print(f"Token success rate:        {total_correct_tokens_v2/total_attempted_tokens_v2*100:.2f}%")
    
    print(f"\n--- Combined Performance ---")
    total_successful_prompts = successful_prompts_v1 + successful_prompts_v2
    print(f"Total successful prompts:  {total_successful_prompts}/{total_prompts_tested}")
    if total_prompts_tested > 0:
        print(f"Combined prompt success:   {total_successful_prompts/total_prompts_tested*100:.2f}%")
    print(f"Total correct tokens:      {total_correct_tokens}/{total_attempted_tokens}")
    if total_attempted_tokens > 0:
        print(f"Combined token success:    {total_correct_tokens/total_attempted_tokens*100:.2f}%")

    print(f"\n--- Timing Statistics ---")
    print(f"Total experiment time:          {experiment_elapsed:.2f} seconds ({experiment_elapsed/60:.2f} minutes)")
    print(f"Avg. time per prompt pair:      {avg_pair_time:.2f} seconds")
    print(f"Avg. peek_one_token time:       {avg_peek_one_token_time:.2f} seconds")
    print(f"Avg. _send_attack_requests time: {avg_attack_request_time:.2f} seconds")

    if pair_times:
        print(f"\n--- Pair Time Details ---")
        print(f"Min pair time:                  {min(pair_times):.2f} seconds")
        print(f"Max pair time:                  {max(pair_times):.2f} seconds")

    print(f"\n--- Cache Interference Analysis ---")
    print(f"V1 vs V2 token success rate:")
    if total_attempted_tokens_v1 > 0 and total_attempted_tokens_v2 > 0:
        v1_rate = total_correct_tokens_v1/total_attempted_tokens_v1*100
        v2_rate = total_correct_tokens_v2/total_attempted_tokens_v2*100
        print(f"  V1: {v1_rate:.2f}%")
        print(f"  V2: {v2_rate:.2f}%")
        print(f"  Difference: {abs(v1_rate - v2_rate):.2f}%")
        
        if abs(v1_rate - v2_rate) > 10:
            print(f"  ⚠️  Significant difference detected - possible cache interference")
        else:
            print(f"  ✓ Similar performance - cache interference minimal")
    
    print("=" * 80 + "\n")
    logger.info("[MAIN] DOUBLE attack process completed.")

def single_victim_experiment():
    peek_count = 3
    max_prompt_count = 20  # Maximum prompt count limit (adjust to desired value)

    # Overall experiment start time
    experiment_start_time = time.time()

    victim_prompts, known_prefixes = load_awesome_chatgpt_prompts("data/prompts.csv")

    # Variables for result aggregation
    total_prompts = len(victim_prompts)
    successful_prompts = 0  # Number of prompts fully successful with peek_count
    total_correct_tokens = 0  # Total number of correct tokens
    total_attempted_tokens = total_prompts * peek_count  # Total number of attempted tokens

    # Variables for time measurement
    all_peek_one_token_times = []  # All peek_one_token call times
    all_attack_request_times = []  # All _send_attack_requests call times
    prompt_times = []  # Time taken per victim prompt

    logger.info(f"[MAIN] Starting attacks on {total_prompts} prompts...")
    logger.info(f"[MAIN] Each prompt will attempt to extract {peek_count} tokens.")

    prompts_tested = 0  # Number of prompts actually tested
    prompts_skipped = 0  # Number of prompts skipped due to errors

    # Execute attacks
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

            # Wrap each prompt processing in individual try-except
            try:
                # Start time for each prompt attack
                prompt_start_time = time.time()

                correct_count, extracted_tokens, peek_times, attack_times = main_single(
                    victim_prompt, known_prefix, peek_count=peek_count
                )

                # End time for each prompt attack
                prompt_elapsed = time.time() - prompt_start_time
                prompt_times.append(prompt_elapsed)

                # Collect time data
                all_peek_one_token_times.extend(peek_times)
                all_attack_request_times.extend(attack_times)

                # Aggregate results
                prompts_tested += 1
                total_correct_tokens += correct_count
                if correct_count == peek_count:
                    successful_prompts += 1
            except Exception as e:
                logger.error(f"[MAIN] Error occurred on prompt {i+1}: {e}")
                logger.info(
                    f"[MAIN] Skipping prompt {i+1} and continuing with next prompt..."
                )
                prompts_skipped += 1
                logger.info("[MAIN] Waiting for 2 minutes before continuing...")
                time.sleep(120)  # Wait briefly before continuing
                continue
    except KeyboardInterrupt:
        print(
            "\n\n[MAIN] ⚠️  KeyboardInterrupt detected! Stopping and aggregating results so far..."
        )
        logger.warning("Attack interrupted by user. Generating partial results.")
    # Attack finished

    # Overall experiment end time
    experiment_elapsed = time.time() - experiment_start_time

    # Print final results (both normal termination and interruption)
    actual_attempted_tokens = prompts_tested * peek_count

    # Calculate time statistics
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
