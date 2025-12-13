from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from typing import List, Tuple

REPO_ID = "unsloth/Llama-3.2-1B-Instruct-GGUF"
# MODEL_FILENAME = "Llama-3.2-1B-Instruct-Q3_K_M.gguf"
MODEL_FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
MODEL_PATH = "meta-llama/Llama-3.2-1B"

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=4096,
    logits_all=True,
    verbose=False,
)


def get_next_token_logprobs(prompt: str, top_k: int = 50) -> List[Tuple[str, float]]:
    output = llm(prompt, max_tokens=1, logprobs=top_k, temperature=0.0, echo=False)
    
    if "choices" in output and len(output["choices"]) > 0:
        choice = output["choices"][0]
        logprobs_data = choice.get("logprobs", {})
        top_logprobs = logprobs_data.get("top_logprobs", [])
        
        if top_logprobs and len(top_logprobs) > 0:
            token_logprobs = top_logprobs[0]
            print(f"[DEBUG TOP_LOGPROBS[0]] Type: {type(token_logprobs)}, Items: {list(token_logprobs.items())[:5]}")
            
            candidates = [(token, logprob) for token, logprob in token_logprobs.items()]
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            return candidates
    
    return []


def get_top_k_candidates(prompt: str, k: int = 50) -> List[str]:
    candidates_with_probs = get_next_token_logprobs(prompt, k)
    
    return [token for token, _ in candidates_with_probs]


def get_bottom_k_dummies(prompt: str, k: int = 50) -> List[str]:
    """Get bottom-k (lowest probability) tokens as dummy tokens"""
    candidates_with_probs = get_next_token_logprobs(prompt, top_k=200)  # Get more tokens to find bottom-k
    
    # Sort by logprob ascending (lowest probability first)
    candidates_with_probs.sort(key=lambda x: x[1])
    
    # Return bottom-k tokens
    return [token for token, _ in candidates_with_probs[:k]]
