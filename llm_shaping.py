# llm_shaping.py
import os
import json
import re
from functools import lru_cache

import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("You must export OPENAI_API_KEY before running this script.")
openai.api_key = os.environ["OPENAI_API_KEY"]

LLM_CHOICE = os.getenv("LLM_CHOICE", "llama-3")  # "gpt-4o-mini", "gpt-3.5-turbo", "llama-3", "phi-2"
PHI_SCALE = float(os.getenv("PHI_SCALE", "1.0"))
ROUND_DIGITS = int(os.getenv("PHI_ROUND_DIGITS", "2"))

# ─────────────────────────────────────────────────────────────
# Local models (llama/phi)
# ─────────────────────────────────────────────────────────────
_llama_pipe = None
if LLM_CHOICE in ("phi-2", "llama-3"):
    model_map = {
        "phi-2": "microsoft/phi-2",
        "llama-3": os.path.expanduser("~/llama-models/Llama-3.2-3B")
    }
    model_name = model_map[LLM_CHOICE]
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    _llama_pipe = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
        return_full_text=False,
    )

# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
_num_regex = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _parse_severity(raw: str) -> float:
    raw = (raw or "").strip()
    try:
        d = json.loads(raw)
        return _clamp01(float(d["severity"]))
    except Exception:
        nums = _num_regex.findall(raw)
        if nums:
            return _clamp01(float(nums[-1]))
    return 0.0

llm_logs = []

# ─────────────────────────────────────────────────────────────
# Main compute_potential
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=100_000)
def compute_potential(window_tuple):
    # Cache key = rounded originals
    window_tuple = tuple(round(float(x), ROUND_DIGITS) for x in window_tuple)
    txt = ", ".join(f"{x:.2f}" for x in window_tuple)

    if LLM_CHOICE.startswith("gpt"):
        # OpenAI GPT call
        prompt = (
            "You are an anomaly rater. "
            "Return ONLY valid JSON {\"severity\": <0..1>}.\n"
            f"Sensor readings: [{txt}]"
        )
        resp = openai.chat.completions.create(
            model=LLM_CHOICE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        raw = resp.choices[0].message.content

    else:
        if _llama_pipe is None:
            raise RuntimeError(f"{LLM_CHOICE} requires a local pipeline but none initialized.")
        prompt = (
            "You are an anomaly rater. "
            "Return ONLY JSON like {\"severity\": 0.xx}.\n"
            f"Sensor readings: [{txt}]"
        )
        raw = _llama_pipe(prompt)[0]["generated_text"]

    score = _parse_severity(raw)
    if PHI_SCALE != 1.0:
        score = _clamp01(score * PHI_SCALE)

    llm_logs.append((window_tuple, score))
    return score

# ─────────────────────────────────────────────────────────────
# Reward shaping
# ─────────────────────────────────────────────────────────────
def shaped_reward(raw_reward, s, s2, gamma):
    phi_s  = compute_potential(tuple(s))
    phi_s2 = compute_potential(tuple(s2))
    total = raw_reward + gamma * phi_s2 - phi_s
    print(f"[DEBUG SHAPING] φ(s)={phi_s:.3f}, φ(s')={phi_s2:.3f}, raw={raw_reward:.3f} → total={total:.3f}")
    return total
