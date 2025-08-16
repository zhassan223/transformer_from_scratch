import numpy as np 
from typing import List, Optional, Union

# Optional torch dependency (used if available for model inference)
try:
    import torch
except Exception:
    torch = None

def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    scaled = logits / float(temperature)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def _nucleus_top_p(probs: np.ndarray, p: float) -> np.ndarray:
    """
    Apply top-p (nucleus) filtering to a probability distribution.
    Keeps the smallest prefix of tokens whose cumulative prob >= p, zeros out the rest, and renormalizes.
    """
    if p >= 1.0:
        return probs
    if p <= 0.0:
        out = np.zeros_like(probs)
        out[np.argmax(probs)] = 1.0
        return out
    # sort by prob desc
    idx = np.argsort(probs)[::-1]
    sorted_probs = probs[idx]
    cum = np.cumsum(sorted_probs)
    # find cutoff index where cumulative >= p
    cutoff_idx = np.searchsorted(cum, p, side="left")
    # zero out everything after cutoff_idx
    sorted_probs[cutoff_idx+1:] = 0.0
    # map back and renormalize
    filtered = np.zeros_like(probs)
    filtered[idx] = sorted_probs
    s = filtered.sum()
    if s <= 0.0 or not np.isfinite(s):
        out = np.zeros_like(probs)
        out[np.argmax(probs)] = 1.0
        return out
    return filtered / s

def decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token: Optional[Union[str, int]] = "<|endoftext|>",
) -> str:
    """
    Generate a completion for `prompt`:
    - Stop when end token is produced (tokenizer.eos_token_id or provided end_token) or after max_new_tokens.
    - Temperature scaling applied before sampling. If temperature == 0, uses greedy (argmax).
    - Top-p (nucleus) sampling applied if top_p < 1.0.
    """
    # start from encoded prompt
    tokens: List[int] = list(tokenizer.encode(prompt))

    # try to get eos id from tokenizer if present
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    # handle case where end_token is int (overrides tokenizer eos)
    end_token_id = end_token if isinstance(end_token, int) else eos_token_id

    # set eval/no grad if using torch
    if torch is not None and hasattr(model, "eval"):
        model.eval()

    for _ in range(max_new_tokens):
        # Prepare input and run model
        if torch is not None:
            inp = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, T]
            with torch.no_grad():
                out = model(inp)
            # some models return (logits, ...)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            # last step logits: [V]
            last_logits = logits[:, -1, :].squeeze(0).detach().cpu().numpy()
        else:
            # Non-torch fallback: support .predict or callable
            raw = model.predict(tokens) if hasattr(model, "predict") else model(tokens)
            raw = np.asarray(raw)
            last_logits = raw[-1] if raw.ndim > 1 else raw

        # Choose next token
        if temperature == 0.0:
            # deterministic greedy
            next_token = int(np.argmax(last_logits))
        else:
            probs = softmax_with_temperature(last_logits, temperature=temperature).astype(np.float64).ravel()
            if top_p < 1.0:
                probs = _nucleus_top_p(probs, top_p)
            # numerical safety
            probs = probs / probs.sum()
            next_token = int(np.random.choice(len(probs), p=probs))

        tokens.append(next_token)

        # Stopping conditions
        if end_token_id is not None and next_token == end_token_id:
            break
        if isinstance(end_token, str) and end_token is not None:
            try:
                if tokenizer.decode([next_token]) == end_token:
                    break
            except Exception:
                pass

    return tokenizer.decode(tokens)
