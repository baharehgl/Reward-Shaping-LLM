# llm_shaping.py
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

LLM_CHOICE = os.getenv("LLM_CHOICE", "llama-3")  # e.g. "gpt-4o-mini", "gpt-4o", "llama-3", "phi-2"
PHI_SCALE = float(os.getenv("PHI_SCALE", "1.0"))       # optional scaling of φ(s)
ROUND_DIGITS = int(os.getenv("PHI_ROUND_DIGITS", "2")) # rounding before caching (to avoid cache collisions)

# ─────────────────────────────────────────────────────────────────────────────
# Local HF pipeline (for llama/phi) if requested
# ─────────────────────────────────────────────────────────────────────────────
_llama_pipe = None
MODEL_NAME = None
if LLM_CHOICE == "phi-2":
    MODEL_NAME = "microsoft/phi-2"
elif LLM_CHOICE.startswith("llama-3"):
    # Change this to your local path or HF model id as needed:
    MODEL_NAME = os.path.expanduser("~/llama-models/Llama-3.2-3B")

if MODEL_NAME:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.float16
    )
    _llama_pipe = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=16,
        temperature=0.0,
        do_sample=False,
        return_full_text=False,  # IMPORTANT: don't echo the prompt
    )

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _build_messages(txt: str):
    """
    Construct strict messages so GPT returns ONLY a JSON object: {"severity": <0..1>}.
    """
    system = (
        "You are an anomaly rater. "
        "Return ONLY valid JSON with a single key 'severity' whose value is a decimal between 0.0 and 1.0. "
        "No prose. No explanation."
    )
    # Few-shot to encourage spread (helps GPT avoid collapsing to 0.0)
    examples = (
        "Examples:\n"
        "  Sensor readings: [0.0, 0.0, 0.0, 0.0, ...] -> {\"severity\": 0.00}\n"
        "  Sensor readings: [0.0, 0.0, 0.0, 5.0, 5.0, ...] -> {\"severity\": 0.75}\n\n"
        "Now rate strictly in JSON:\n"
        f"Sensor readings: [{txt}]"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": examples},
    ]

_num_regex = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _parse_severity(raw: str) -> float:
    """
    Prefer strict JSON. If that fails, take the LAST number in the string
    (avoids grabbing the '0.0' from the '0.0–1.0' scale text).
    """
    raw = raw.strip()
    # Try JSON
    try:
        d = json.loads(raw)
        val = float(d["severity"])
        return _clamp01(val)
    except Exception:
        pass
    # Fallback: last number
    nums = _num_regex.findall(raw)
    if not nums:
        raise ValueError(f"Could not extract number from LLM output: {raw!r}")
    return _clamp01(float(nums[-1]))

llm_logs = []

# ─────────────────────────────────────────────────────────────────────────────
# LLM-backed potential φ(s)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=100_000)
def compute_potential(window_tuple):
    """
    Returns φ(s) ∈ [0,1] for a window s using an LLM.
    Uses strict JSON for GPT models; robust parsing fallback otherwise.
    """
    # Round to reduce cache key explosion but keep enough resolution
    window_tuple = tuple(round(float(x), ROUND_DIGITS) for x in window_tuple)
    txt = ", ".join(f"{x:.2f}" for x in window_tuple)

    if LLM_CHOICE.startswith("gpt"):
        messages = _build_messages(txt)
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
        # Instruct local model to output JSON too
        prompt = (
            "You are an anomaly rater. "
            "Return ONLY valid JSON: {\"severity\": <0..1>} with no extra text.\n"
            f"Sensor readings: [{txt}]"
        )
        raw = _llama_pipe(prompt)[0]["generated_text"]

    score = _parse_severity(raw)

    # Optional scaling (if you find GPT's range too narrow)
    if PHI_SCALE != 1.0:
        score = _clamp01(score * PHI_SCALE)

    llm_logs.append((window_tuple, score))
    return score

# ─────────────────────────────────────────────────────────────────────────────
# Potential-based shaping: r' = r + γ φ(s') − φ(s)
# ─────────────────────────────────────────────────────────────────────────────
def shaped_reward(raw_reward, s, s2, gamma):
    φ_s  = compute_potential(tuple(s))
    φ_s2 = compute_potential(tuple(s2))
    total = raw_reward + gamma * φ_s2 - φ_s
    print(f"[DEBUG SHAPING] φ(s)={φ_s:.3f}, φ(s')={φ_s2:.3f}, raw={raw_reward:.3f} → total={total:.3f}")
    return total