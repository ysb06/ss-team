import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
import torch.nn.functional as F

MODEL_ID = "meta-llama/Llama-3.2-1B"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def get_next_token_logprobs(prompt: str, top_k: int = 50) -> List[Tuple[str, float]]:
    """Get next token candidates with their log probabilities"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]  # Get the token IDs
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Get logits for next token
        
        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get top-k tokens
        top_k_logprobs, top_k_indices = torch.topk(log_probs, k=min(top_k, len(log_probs)))
        
        # Convert to list of (token, logprob) tuples
        candidates = []
        for logprob, idx in zip(top_k_logprobs.tolist(), top_k_indices.tolist()):
            # IMPORTANT: Decode by appending the new token to the original sequence
            # This preserves the tokenizer's context (spaces, subword boundaries, etc.)
            new_ids = torch.cat([input_ids, torch.tensor([idx]).to(input_ids.device)])
            full_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            
            # Extract just the new token by removing the original prompt
            new_token = full_text[len(prompt):]
            candidates.append((new_token, logprob))
        
        print(f"[DEBUG TOP_LOGPROBS[0]] Type: list, Items: {candidates[:5]}")
        
        return candidates


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
