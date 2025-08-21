import os, json, re, math
from functools import lru_cache
import numpy as np
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---------------- Config ----------------
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("You must export OPENAI_API_KEY")
openai.api_key = os.environ["OPENAI_API_KEY"]

LLM_CHOICE   = os.getenv("LLM_CHOICE", "phi-2")   # "gpt-4o(-mini)", "gpt-3.5-turbo", "llama-3", "phi-2"
PHI_SCALE    = float(os.getenv("PHI_SCALE", "1.0"))
ROUND_DIGITS = int(os.getenv("PHI_ROUND_DIGITS", "2"))
PHI_DEBUG    = os.getenv("PHI_DEBUG", "0") == "1"       # prints a few raw responses

# ------------- Local pipelines -------------
_llama_pipe = None
if LLM_CHOICE in ("llama-3", "phi-2"):
    model_map = {
        "llama-3": os.path.expanduser("~/llama-models/Llama-3.2-3B"),
        "phi-2":   "microsoft/phi-2",
    }
    model_name = model_map[LLM_CHOICE]
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    _llama_pipe = pipeline(
        "text-generation", model=mdl, tokenizer=tok,
        max_new_tokens=32, temperature=0.0, do_sample=False, return_full_text=False
    )

# ------------- Utilities -------------
_num_regex = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _clamp01(x: float) -> float: return max(0.0, min(1.0, x))

def _normalize_for_prompt(xs):
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0: return xs.tolist()
    mu = xs.mean()
    sd = xs.std()
    return ([0.0] * len(xs)) if sd < 1e-8 else ((xs - mu) / sd).tolist()

def _build_prompt(txt: str) -> str:
    return (
        "Task: Score anomaly severity in a numeric sequence.\n"
        "Return ONLY JSON {\"severity\": s} with s in [0.0, 1.0]. Use the full range.\n\n"
        "Guidelines:\n"
        " • 0.00 → purely normal noise or repeating/periodic oscillations.\n"
        " • 0.25 → small drift or very mild irregularity.\n"
        " • 0.50 → clear mean shift or occasional unusual spikes.\n"
        " • 0.75 → sustained deviation or clusters of large spikes.\n"
        " • 0.95 → extreme outliers or breaks in the overall pattern.\n\n"
        "Important:\n"
        " • Do NOT flag repeating peaks, oscillations, or regular periodic patterns as anomalies.\n"
        " • Only raise the score if the sequence departs from its usual pattern or shows sustained abnormal behavior.\n"
        " • Keep normal sections near 0.0.\n\n"
        "Examples:\n"
        "  [0,0,0,0,0]             -> {\"severity\": 0.00}\n"
        "  [0,0,0.1,0.2,0.1]       -> {\"severity\": 0.25}\n"
        "  [0,0,1.5,1.7,1.6]       -> {\"severity\": 0.70}\n"
        "  [0,3.0,0,-2.5,0]        -> {\"severity\": 0.95}\n"
        f"\nSequence: [{txt}]"
    )
def _parse_severity(raw: str) -> float:
    raw = (raw or "").strip()
    # JSON first
    try:
        d = json.loads(raw)
        return _clamp01(float(d["severity"]))
    except Exception:
        pass
    # last number fallback
    nums = _num_regex.findall(raw)
    return _clamp01(float(nums[-1])) if nums else 0.0

def _heuristic_phi(window_tuple):
    """Only used if the LLM returns nothing parseable, to avoid all-zero logs."""
    xs = np.asarray(window_tuple, dtype=float)
    if xs.size == 0: return 0.0
    mu, sd = xs.mean(), xs.std()
    if sd < 1e-8: return 0.0
    z_last = abs(xs[-1] - mu) / (sd + 1e-8)
    # squash into [0,1] with a smooth curve: ~0 at z=0, ~0.5 at z≈1.5, ~0.9 at z≈3
    return _clamp01(1.0 - math.exp(-0.5 * z_last ** 2))

llm_logs = []

# ------------- Main -------------
@lru_cache(maxsize=100_000)
def compute_potential(window_tuple):
    # cache key uses rounded originals
    window_tuple = tuple(round(float(x), ROUND_DIGITS) for x in window_tuple)

    # prompt uses z-scored values for better relative visibility
    z = _normalize_for_prompt(window_tuple)
    txt = ", ".join(f"{v:.2f}" for v in z)
    prompt = _build_prompt(txt)

    raw = ""
    try:
        if LLM_CHOICE.startswith("gpt"):
            resp = openai.chat.completions.create(
                model=LLM_CHOICE,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=16,
            )
            raw = resp.choices[0].message.content
        else:
            if _llama_pipe is None:
                raise RuntimeError(f"{LLM_CHOICE} requires a local HF pipeline.")
            raw = _llama_pipe(prompt)[0]["generated_text"]
    except Exception as e:
        # If API/pipeline fails, keep raw="" so we fall back to heuristic
        if PHI_DEBUG:
            print("[PHI DEBUG] LLM call failed:", e)

    score = _parse_severity(raw)
    if score == 0.0 and not raw:
        # only when the model gave us nothing parsable
        score = _heuristic_phi(window_tuple)

    if PHI_SCALE != 1.0:
        score = _clamp01(score * PHI_SCALE)

    if PHI_DEBUG:
        print(f"[PHI DEBUG] raw={raw!r} -> score={score:.3f}")
    llm_logs.append((window_tuple, score))
    return score

def shaped_reward(raw_reward, s, s2, gamma):
    phi_s  = compute_potential(tuple(s))
    phi_s2 = compute_potential(tuple(s2))
    total = raw_reward + gamma * phi_s2 - phi_s
    print(f"[DEBUG SHAPING] φ(s)={phi_s:.3f}, φ(s')={phi_s2:.3f}, raw={raw_reward:.3f} → total={total:.3f}")
    return total