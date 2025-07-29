# llm_shaping.py

import os, re, numpy as np, openai
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

openai.api_key = os.getenv("OPENAI_API_KEY", "")
LLM_CHOICE   = os.getenv("LLM_CHOICE",    "gpt-3.5-turbo")

# Prepare llama pipeline if needed...
_llama_pipe = None
if LLM_CHOICE.startswith("llama-3"):
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70b-chat")
    mdl = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-70b-chat", device_map="auto", torch_dtype="auto"
    )
    _llama_pipe = pipeline("text-generation", model=mdl, tokenizer=tok,
                            max_new_tokens=8, temperature=0.0, do_sample=False)

@lru_cache(maxsize=None)
def _compute_api(window_tuple):
    """Call the LLM once to get φ(window)."""
    txt    = ", ".join(f"{x:.2f}" for x in window_tuple)
    prompt = f"Sensor readings: [{txt}]\nRate severity from 0.0 to 1.0, return only the number."
    if LLM_CHOICE.startswith("gpt"):
        resp = openai.chat.completions.create(
            model=LLM_CHOICE,
            messages=[{"role":"user","content":prompt}],
            temperature=0, max_tokens=4,
        )
        raw = resp.choices[0].message.content.strip()
    else:
        raw = _llama_pipe(prompt)[0]["generated_text"].strip()
    m = re.search(r"([0-9]*\\.?[0-9]+)", raw)
    val = float(m.group(1)) if m else 0.0
    return max(0.0, min(1.0, val))

# This is the function your RL loop will use,
# but we'll monkey-patch it at runtime:
PHI_LOOKUP = {}
llm_logs    = []

def compute_potential(window_tuple):
    key = tuple(np.round(window_tuple, 2))
    phi = PHI_LOOKUP.get(key, 0.0)
    llm_logs.append((window_tuple, phi))
    return phi

def shaped_reward(raw_reward, s, s2, gamma):
    φ_s  = compute_potential(tuple(s))
    φ_s2 = compute_potential(tuple(s2))
    return raw_reward + gamma * φ_s2 - φ_s