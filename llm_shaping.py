# llm_shaping.py
import os
from functools import lru_cache
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

openai.api_key = os.getenv("OPENAI_API_KEY")
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

@lru_cache(maxsize=10000)
def compute_potential(window_tuple):
    txt = ", ".join(f"{x:.2f}" for x in window_tuple)
    prompt = f"Sensor readings: [{txt}]\nRate severity from 0.0 (normal) to 1.0 (critical)."
    if LLM_CHOICE.startswith("gpt"):
        r = openai.ChatCompletion.create(
            model=LLM_CHOICE,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0, max_tokens=4
        )
        score = float(r.choices[0].message.content.strip())
    else:
        out = _llama_pipe(prompt)[0]["generated_text"]
        score = float(out.strip().split()[-1])
    score = max(0.0, min(1.0, score))
    llm_logs.append((window_tuple, score))
    return score

def shaped_reward(raw_reward, s, s2, gamma):
    φ_s  = compute_potential(tuple(s))
    φ_s2 = compute_potential(tuple(s2))
    llm_logs.append((s.copy(), φ_s))
    return raw_reward + gamma * φ_s2 - φ_s