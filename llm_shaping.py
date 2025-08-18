# llm_shaping.py  (drop-in replacement)
import os
import json
import re
import openai
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("You must export OPENAI_API_KEY before running this script.")
openai.api_key = os.environ["OPENAI_API_KEY"]

# e.g. "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "llama-3", "phi-2"
LLM_CHOICE = os.getenv("LLM_CHOICE", "llama-3")

# Optional tuning knobs
PHI_SCALE    = float(os.getenv("PHI_SCALE", "1.0"))        # scales φ, clamped to [0,1]
ROUND_DIGITS = int(os.getenv("PHI_ROUND_DIGITS", "2"))     # cache key rounding
MAX_NEW_TOK  = int(os.getenv("PHI_MAX_NEW_TOKENS", "32"))  # headroom for local gen

# ─────────────────────────────────────────────────────────────────────────────
# Local HF pipeline (for llama/phi) if requested
# ─────────────────────────────────────────────────────────────────────────────
_llama_pipe = None
MODEL_NAME = None
if LLM_CHOICE == "phi-2":
    MODEL_NAME = "microsoft/phi-2"
elif LLM_CHOICE.startswith("llama-3"):
    # Change this to your local path or a HF model id (must be Instruct/Chat variant)
    MODEL_NAME = os.path.expanduser("~/llama-models/Llama-3.2-3B")

tok = None
mdl = None
if MODEL_NAME:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.float16
    )
    _llama_pipe = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=MAX_NEW_TOK,
        temperature=0.0,
        do_sample=False,
        return_full_text=False,
        eos_token_id=tok.eos_token_id
    )

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
_num_regex = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _normalize_for_prompt(xs):
    """Z-score normalize only for the prompt (does not change cache keys or math)."""
    xs = list(xs)
    if not xs:
        return xs
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    sd = var ** 0.5
    if sd < 1e-8:
        return [0.0 for _ in xs]
    return [(x - mu) / sd for x in xs]

def _build_messages(txt: str):
    """
    Strict messages so GPT returns ONLY JSON { "severity": <0..1> }.
    Few-shot covers low/med/high; discourages collapsing to 0.0.
    """
    system = (
        "You are an anomaly rater. "
        "Return ONLY valid JSON: {\"severity\": <0..1>}. "
        "Use the full range; avoid always returning 0.0 for subtle shifts."
    )
    examples = (
        "Examples:\n"
        "  Sensor readings: [0.0, 0.0, 0.0, 0.0, ...] -> {\"severity\": 0.00}\n"
        "  Sensor readings: [0.0, 0.0, 0.1, 0.2, 0.1, ...] -> {\"severity\": 0.25}\n"
        "  Sensor readings: [0.0, 0.0, 1.5, 1.7, 1.6, ...] -> {\"severity\": 0.70}\n"
        "  Sensor readings: [0.0, 3.0, 0.0, -2.5, 0.0, ...] -> {\"severity\": 0.95}\n\n"
        f"Now rate strictly in JSON for: [{txt}]"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": examples},
    ]

def _parse_severity(raw: str) -> float:
    """
    Prefer strict JSON. If that fails, take the LAST number in the string.
    No crashes: final fallback = 0.0.
    """
    raw = (raw or "").strip()
    if raw:
        # Try JSON
        try:
            d = json.loads(raw)
            val = float(d["severity"])
            return _clamp01(val)
        except Exception:
            pass
        # Fallback: last number in text
        nums = _num_regex.findall(raw)
        if nums:
            return _clamp01(float(nums[-1]))
    # Final fallback
    return 0.0

def _llama_generate_json(txt: str) -> str:
    """
    Generate JSON using Llama with proper chat template and a retry path.
    """
    assert tok is not None and mdl is not None and _llama_pipe is not None

    system = (
        "You are an anomaly rater. "
        "Return ONLY valid JSON with a single key 'severity' whose value is a decimal between 0.0 and 1.0. "
        "No prose. No explanation."
    )
    user = f"Sensor readings: [{txt}]"

    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Try with return_full_text=False
    out = _llama_pipe(prompt)
    raw = (out[0].get("generated_text", "") or "").strip()
    if raw:
        return raw

    # Retry with return_full_text=True
    pipe_full = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=MAX_NEW_TOK,
        temperature=0.0,
        do_sample=False,
        return_full_text=True,
        eos_token_id=tok.eos_token_id
    )
    out_full = pipe_full(prompt)
    raw_full = (out_full[0].get("generated_text", "") or "").strip()
    if raw_full:
        # If template markers exist, take tail; otherwise return as-is
        return raw_full.split("</s>")[-1].strip()

    # Safe fallback
    return '{"severity": 0.0}'

llm_logs = []

# ─────────────────────────────────────────────────────────────────────────────
# LLM-backed potential φ(s)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=100_000)
def compute_potential(window_tuple):
    """
    Returns φ(s) ∈ [0,1] for a window s using an LLM.
    - Cache key uses rounded ORIGINAL window.
    - Prompt uses z-score normalized values for better dynamic range.
    """
    # 1) Cache key: rounded originals (stable)
    window_tuple = tuple(round(float(x), ROUND_DIGITS) for x in window_tuple)

    # 2) Prompt: normalized (z-score), improves sensitivity to relative changes
    norm_for_prompt = _normalize_for_prompt(window_tuple)
    txt = ", ".join(f"{x:.2f}" for x in norm_for_prompt)

    # 3) Call the chosen LLM
    if LLM_CHOICE.startswith("gpt"):
        messages = _build_messages(txt)
        # Prefer JSON mode (supported by GPT-4o family); fall back if not available
        try:
            resp = openai.chat.completions.create(
                model=LLM_CHOICE,
                messages=messages,
                temperature=0.0,
                max_tokens=16,
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = openai.chat.completions.create(
                model=LLM_CHOICE,
                messages=messages,
                temperature=0.0,
                max_tokens=16,
            )
        raw = resp.choices[0].message.content
    else:
        if _llama_pipe is None:
            raise RuntimeError(
                f"LLM_CHOICE={LLM_CHOICE!r} requires a local HF pipeline, but none is initialized."
            )
        raw = _llama_generate_json(txt)

    # 4) Parse and scale
    score = _parse_severity(raw)
    if PHI_SCALE != 1.0:
        score = _clamp01(score * PHI_SCALE)

    llm_logs.append((window_tuple, score))
    return score

# ─────────────────────────────────────────────────────────────────────────────
# Potential-based shaping: r' = r + γ φ(s') − φ(s)
# ─────────────────────────────────────────────────────────────────────────────
def shaped_reward(raw_reward, s, s2, gamma):
    phi_s  = compute_potential(tuple(s))
    phi_s2 = compute_potential(tuple(s2))
    total = raw_reward + gamma * phi_s2 - phi_s
    print(f"[DEBUG SHAPING] φ(s)={phi_s:.3f}, φ(s')={phi_s2:.3f}, raw={raw_reward:.3f} → total={total:.3f}")
    return total
