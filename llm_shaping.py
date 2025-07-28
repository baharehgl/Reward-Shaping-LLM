# llm_shaping.py
import os, re, pickle
import numpy as np
import openai
from functools import lru_cache
import torch

# These imports only used when we actually call the API:
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -----------------------------------------------------------------------------
# Global state: will be filled by train_wrapper's precompute pass.
# -----------------------------------------------------------------------------
PHI_LOOKUP = {}      # window_tuple → φ
llm_logs    = []     # to replay for CSV after training

# Which model are we using right now?
LLM_CHOICE = os.getenv("LLM_CHOICE", "gpt-3.5-turbo")
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

# If using Llama-3, prepare a local pipeline:
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

# Live-API path: we only use this when PHI_LOOKUP is empty (i.e. on first pass)
@lru_cache(maxsize=100_000)
def _compute_api(window_tuple):
    txt = ", ".join(f"{x:.2f}" for x in window_tuple)
    prompt = f"Sensor readings: [{txt}]\nRate severity from 0.0 to 1.0, return only a number."
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
    val = float(m.group(1)) if m else 0.0
    return max(0.0, min(1.0, val))

def compute_potential(window_tuple):
    """
    If PHI_LOOKUP is not yet populated, call the API.
    Once PHI_LOOKUP has been filled (by your train_wrapper), this will
    do a fast dict lookup only.
    """
    # Quantize key
    key = tuple(np.round(window_tuple, 2))

    if PHI_LOOKUP:
        # lookup‐mode
        phi = PHI_LOOKUP.get(key, 0.0)
    else:
        # build‐mode → call the API once
        phi = _compute_api(key)

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