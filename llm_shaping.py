# llm_shaping.py

import openai
from functools import lru_cache

openai.api_key = "<YOUR_API_KEY>"

@lru_cache(maxsize=10000)
def compute_potential(window_tuple):
    prompt = (
        f"Sensor readings: {', '.join(map(str, window_tuple))}\n"
        "Rate from 0.0 (normal) to 1.0 (critical fault) how severe this is."
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=4
    )
    return float(resp.choices[0].message.content.strip())

def shaped_reward(raw_reward, s, s2, gamma):
    φ_s  = compute_potential(tuple(s))
    φ_s2 = compute_potential(tuple(s2))
    return raw_reward + gamma * φ_s2 - φ_s
