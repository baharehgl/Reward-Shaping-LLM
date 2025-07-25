# llm_shaping.py
import os
import pickle
import numpy as np

BUILD_LOOKUP = os.getenv("BUILD_LOOKUP", "0") == "1"
model_name  = os.getenv("LLM_CHOICE", "gpt-3.5-turbo")
lookup_fn   = os.path.join(os.path.dirname(__file__),
                           f"phi_lookup_{model_name}.pkl")

llm_logs = []

if BUILD_LOOKUP:
    # ----------------------------------------------------
    # Build-mode: expose the original API-powered compute
    # ----------------------------------------------------
    import openai, re
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    from functools import lru_cache

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY to run build_phi_lookup.py")

    openai.api_key = os.environ["OPENAI_API_KEY"]
    LLM_CHOICE = model_name

    _llama_pipe = None
    if LLM_CHOICE.startswith("llama-3"):
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70b-chat")
        mdl = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3-70b-chat", device_map="auto", torch_dtype=torch.float16
        )
        _llama_pipe = pipeline(
            "text-generation", model=mdl, tokenizer=tok,
            max_new_tokens=8, temperature=0.0, do_sample=False
        )

    @lru_cache(maxsize=50_000)
    def compute_potential(window_tuple):
        txt = ", ".join(f"{x:.2f}" for x in window_tuple)
        prompt = f"Sensor readings: [{txt}]\nRate severity from 0.0 to 1.0, respond with a single numeric value."
        if LLM_CHOICE.startswith("gpt"):
            resp = openai.ChatCompletion.create(
                model=LLM_CHOICE,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
                max_tokens=4,
            )
            raw = resp.choices[0].message.content.strip()
        else:
            raw = _llama_pipe(prompt)[0]["generated_text"].strip()
        m = re.search(r"([0-9]*\.?[0-9]+)", raw)
        score = float(m.group(1)) if m else 0.0
        score = max(0.0, min(1.0, score))
        return score

else:
    # ----------------------------------------------------
    # Lookup-mode: load precomputed phi_lookup_<model>.pkl
    # ----------------------------------------------------
    if not os.path.exists(lookup_fn):
        raise FileNotFoundError(
            f"No lookup for {model_name}; run:\n"
            f"  BUILD_LOOKUP=1 LLM_CHOICE={model_name} python build_phi_lookup.py"
        )
    with open(lookup_fn, "rb") as f:
        PHI_LOOKUP = pickle.load(f)

    def compute_potential(window_tuple):
        key = tuple(np.round(window_tuple, 2))
        phi = PHI_LOOKUP.get(key, 0.0)
        llm_logs.append((window_tuple, phi))
        return phi

def shaped_reward(raw_reward, s, s2, gamma):
    φ_s  = compute_potential(tuple(s))
    φ_s2 = compute_potential(tuple(s2))
    return raw_reward + gamma * φ_s2 - φ_s


'''
import os
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
from functools import lru_cache

# -----------------------------------------------------------------------------
# 1) Make sure the key is set, or die loudly
# -----------------------------------------------------------------------------
if "OPENAI_API_KEY" not in os.environ:
    print(">>> USING API KEY:", os.environ.get("OPENAI_API_KEY")[:5] + "…")
    raise RuntimeError("You must export OPENAI_API_KEY before running this script.")
openai.api_key = os.environ["OPENAI_API_KEY"]

LLM_CHOICE = os.getenv("LLM_CHOICE", "gpt-3.5-turbo")  # or "gpt-4-0613" or "llama-3"

# If using Llama-3:
_llama_pipe = None
if LLM_CHOICE.startswith("llama-3"):
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70b-chat")
    mdl = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-70b-chat", device_map="auto", torch_dtype=torch.float16
    )
    _llama_pipe = pipeline(
        "text-generation", model=mdl, tokenizer=tok,
        max_new_tokens=8, temperature=0.0, do_sample=False
    )

llm_logs = []
@lru_cache(maxsize=10_000)
def compute_potential(window_tuple):
    txt = ", ".join(f"{x:.2f}" for x in window_tuple)
    prompt = (
        f"Sensor readings: [{txt}]\n"
        "Rate severity from 0.0 (normal) to 1.0 (critical). "
        "Respond with only a single numeric value between 0.0 and 1.0, no extra text."
    )
    print(f"[LLM CALL] model={LLM_CHOICE!r}  prompt='{prompt[:60]}…'")

    if LLM_CHOICE.startswith("gpt"):
        print(f"[LLM CALL] model={LLM_CHOICE!r} prompt={prompt!r}")
        resp = openai.chat.completions.create(
            model=LLM_CHOICE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,
        )
        raw = resp.choices[0].message.content.strip()
        print(f"[LLM RAW ] {raw!r}")
    else:
        raw = _llama_pipe(prompt)[0]["generated_text"].strip()

    # extract the first floating‐point number we see
    m = re.search(r"([0-9]*\.?[0-9]+)", raw)
    if not m:
        raise ValueError(f"Could not extract a number from LLM output: {raw!r}")
    score = float(m.group(1))

    # clamp to [0,1]
    score = max(0.0, min(1.0, score))
    llm_logs.append((window_tuple, score))
    return score
def shaped_reward(raw_reward, s, s2, gamma):
    φ_s  = compute_potential(tuple(s))
    φ_s2 = compute_potential(tuple(s2))
    total = raw_reward + gamma * φ_s2 - φ_s
    print(f"[DEBUG SHAPING] φ(s)={φ_s:.3f}, φ(s')={φ_s2:.3f}, raw={raw_reward:.3f} → total={total:.3f}")
    # note: no need for a second append here, compute_potential already logged both φ(s) and φ(s2)
    return total
'''