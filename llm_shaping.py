# llm_shaping.py
import os
import json
import re
from functools import lru_cache

import numpy as np
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("You must export OPENAI_API_KEY before running this script.")
openai.api_key = os.environ["OPENAI_API_KEY"]

LLM_CHOICE  = os.getenv("LLM_CHOICE", "gpt-3.5-turbo")  # e.g. "gpt-4o", "llama-3", "phi-2"
PHI_SCALE   = float(os.getenv("PHI_SCALE", "1.0"))      # optional scaling of φ(s)
ROUND_DIGITS = int(os.getenv("PHI_ROUND_DIGITS", "2"))  # rounding before caching

# ─────────────────────────────────────────────────────────────────────────────
# Local HF pipeline (for llama/phi) if requested
# ─────────────────────────────────────────────────────────────────────────────
_llama_pipe = None
MODEL_NAME = None
if LLM_CHOICE == "phi-2":
    MODEL_NAME = "microsoft/phi-2"
elif LLM_CHOICE.startswith("llama-3"):
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
        return_full_text=False,  # don't echo the prompt
    )

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
_num_regex = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _parse_severity(raw: str) -> float:
    """Strict JSON first, else fallback to the LAST number (avoids grabbing '0.0' from '0.0–1.0')."""
    raw = raw.strip()
    try:
        d = json.loads(raw)
        return _clamp01(float(d["severity"]))
    except Exception:
        nums = _num_regex.findall(raw)
        if not nums:
            raise ValueError(f"Could not extract number from LLM output: {raw!r}")
        return _clamp01(float(nums[-1]))

def _to_nested_tuple(win) -> tuple:
    """
    Make a hashable nested tuple with rounding.
      - 1D input -> (v1, v2, ...)
      - 2D input (n_steps, 2) -> ((v1,f1), (v2,f2), ...)
    """
    arr = np.asarray(win)
    if arr.ndim == 1:
        arr = np.round(arr.astype(float), ROUND_DIGITS)
        return tuple(float(x) for x in arr.tolist())
    elif arr.ndim == 2 and arr.shape[1] == 2:
        arr = np.round(arr.astype(float), ROUND_DIGITS)
        return tuple((float(v), float(f)) for v, f in arr.tolist())
    else:
        arr = np.round(arr.astype(float).flatten(), ROUND_DIGITS)
        return tuple(float(x) for x in arr.tolist())

def _format_window_for_prompt(nested: tuple) -> str:
    """Pretty string: '(v,f)' when 2D, else 'v' comma-separated."""
    if len(nested) and isinstance(nested[0], tuple) and len(nested[0]) == 2:
        return ", ".join(f"({v:.2f},{int(f)})" for (v, f) in nested)
    return ", ".join(f"{float(x):.2f}" for x in nested)

def _build_messages_for_gpt(txt: str):
    system = (
        "You are an anomaly rater. "
        "Return ONLY valid JSON with a single key 'severity' whose value is a decimal in [0.0, 1.0]. "
        "No prose. No explanation."
    )
    user = (
        "Rate anomaly severity of the sliding window on a 0.0–1.0 scale.\n"
        "Each item is (value,flag), where flag reflects the agent's chosen label at that step (0=normal, 1=anomaly).\n"
        "Output EXACTLY: {\"severity\": number}.\n\n"
        "Examples:\n"
        "  Window: [(0.00,0), (0.00,0), (0.00,0), (0.00,0)] -> {\"severity\": 0.00}\n"
        "  Window: [(0.00,0), (0.00,0), (4.50,1), (4.80,1)] -> {\"severity\": 0.75}\n\n"
        f"Window: [{txt}]"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

llm_logs = []

# ─────────────────────────────────────────────────────────────────────────────
# LLM-backed potential φ(s) — cached on a normalized, hashable key
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=100_000)
def _compute_potential_cached(nested_window: tuple) -> float:
    txt = _format_window_for_prompt(nested_window)

    if LLM_CHOICE.startswith("gpt"):
        messages = _build_messages_for_gpt(txt)
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
        prompt = (
            "You are an anomaly rater. "
            "Return ONLY valid JSON: {\"severity\": <0..1>} with no extra text.\n"
            "Each item is (value,flag), where flag is 0=normal, 1=anomaly.\n"
            f"Window: [{txt}]"
        )
        raw = _llama_pipe(prompt)[0]["generated_text"]

    score = _parse_severity(raw)
    if PHI_SCALE != 1.0:
        score = _clamp01(score * PHI_SCALE)

    llm_logs.append((nested_window, score))
    return score

def compute_potential(window) -> float:
    """Public entry-point. Accepts 1D values or 2D (value,flag) windows."""
    nested = _to_nested_tuple(window)
    return _compute_potential_cached(nested)

# ── Backward compatibility: expose .cache_clear() / .cache_info() on compute_potential
def _compat_cache_clear():
    _compute_potential_cached.cache_clear()

def _compat_cache_info():
    return _compute_potential_cached.cache_info()

compute_potential.cache_clear = _compat_cache_clear
compute_potential.cache_info  = _compat_cache_info

def clear_phi_cache():
    """Explicit helper your code can call instead of touching .cache_clear()."""
    _compute_potential_cached.cache_clear()
    # also clear any logs if you usually do that alongside
    # (comment out if you want to keep the logs)
    # llm_logs.clear()

# ─────────────────────────────────────────────────────────────────────────────
# Potential-based shaping: r' = r + γ φ(s') − φ(s)
# ─────────────────────────────────────────────────────────────────────────────
def shaped_reward(raw_reward, s, s2, gamma):
    """
    s  : current window (1D values or 2D (value,flag))
    s2 : action-specific next window (must differ in the last flag for action 0 vs 1)
    """
    phi_s  = compute_potential(s)
    phi_s2 = compute_potential(s2)
    total = raw_reward + gamma * phi_s2 - phi_s
    print(f"[DEBUG SHAPING] φ(s)={phi_s:.3f}, φ(s')={phi_s2:.3f}, raw={raw_reward:.3f} → total={total:.3f}")
    return total