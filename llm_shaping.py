# llm_shaping.py

import os
from functools import lru_cache

# --- OpenAI client ---
import openai

# --- HuggingFace client for Llama 3 ---
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    set_seed
)

# Read your OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Choose your model here or override via env var
LLM_CHOICE = os.getenv("LLM_CHOICE", "gpt-3.5-turbo")
# If you want GPT-4: set LLM_CHOICE="gpt-4-0613"
# If you want Llama 3: set LLM_CHOICE="llama-3"

# Set up a HF pipeline if Llama 3
_llama_pipeline = None
if LLM_CHOICE.startswith("llama-3"):
    # replace with the exact name of your Llama 3 model on HF
    _tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70b-chat")
    _model     = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-70b-chat",
        device_map="auto", torch_dtype="auto"
    )
    _llama_pipeline = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
        max_length=50,
        temperature=0.0,
        do_sample=False
    )

# For histogram / post‐plotting
llm_logs = []

@lru_cache(maxsize=10000)
def compute_potential(window_tuple):
    """Return Φ(s) ∈ [0,1] for a sliding window of floats."""
    text = ", ".join(f"{x:.2f}" for x in window_tuple)
    prompt = (
        f"Given these sensor readings:\n[{text}]\n"
        "Rate from 0.0 (normal) to 1.0 (critical fault) how severe this is."
    )

    if LLM_CHOICE.startswith("gpt"):
        # OpenAI call
        resp = openai.ChatCompletion.create(
            model=LLM_CHOICE,
            messages=[{"role":"user","content": prompt}],
            temperature=0.0,
            max_tokens=4
        )
        score = float(resp.choices[0].message.content.strip())
    else:
        # Llama 3 via HF pipeline
        out = _llama_pipeline(prompt)[0]["generated_text"]
        # assume it ends with a number like "… 0.78"
        score = float(out.strip().split()[-1])

    score = max(0.0, min(1.0, score))
    llm_logs.append((window_tuple, score))
    return score

def shaped_reward(raw_reward, s, s2, gamma):
    """Potential‐based shaping: R' = R + γΦ(s') − Φ(s)."""
    phi_s  = compute_potential(tuple(s))
    phi_s2 = compute_potential(tuple(s2))
    return raw_reward + gamma * phi_s2 - phi_s