# Reward-Shaping-LLM
**LLM-based potential reward shaping for RL time-series anomaly detection (with VAE dynamic scaling + active learning).**  
LSTM-DQN agent + LLM-derived semantic potentials + VAE reconstruction guidance + active learning/label propagation.

[📄 Paper (PDF)](./paper/IEEE_Conference__LLM_Based_Potential_Reward.pdf) 

---

## ✨ What’s inside
- **LLM-based Potential-Based Reward Shaping (PBRS)**  
  Policy-invariant shaping with a potential \( \phi(s) \) predicted by an LLM over sliding windows:  
  \[
  r'(s,a,s') = r(s,a,s') + \gamma \,\phi(s') - \phi(s)
  \]
- **Dynamic reward blending**  
  Mix supervised classification feedback with VAE reconstruction error using a time-varying controller \( \lambda(t) \).
- **Active learning + label propagation**  
  Query the most uncertain windows (small Q-margin) and propagate labels to neighbors.
- **Benchmarks**  
  Yahoo-A1 (univariate) and SMD (multivariate) under limited labels.

---

## 🖼️ Method at a glance
An LSTM-based RL agent acts on sliding windows; the reward merges (i) classification signal, (ii) VAE reconstruction error, and (iii) an LLM-derived PBRS potential. An active-learning module requests labels where the agent is most uncertain.

---

## 📦 Installation
```bash
# 1) Clone
git clone https://github.com/baharehgl/Reward-Shaping-LLM.git
cd Reward-Shaping-LLM

# 2) (Optional) Create a fresh environment
conda create -y -n rsl python=3.10
conda activate rsl

# 3) Install
pip install -r requirements.txt
# If you use OpenAI or other providers, also: pip install openai


## 🔧 Configuration
```
seed: 42
device: "cuda"   # or "cpu"

data:
  name: "yahoo_a1"   # or "smd"
  window: 25
  stride: 1

rl:
  gamma: 0.99
  algo: "dqn"
  hidden_size: 128

rewards:
  # Confusion-matrix reward (TP/TN/FP/FN)
  tp: 5
  tn: 1
  fp: -1
  fn: -5
  # VAE dynamic scaling
  lambda:
    init: 0.2
    alpha: 0.05
    target_episode_reward: 50
    min: 0.0
    max: 1.0

vae:
  latent_dim: 16
  hidden_size: 128

llm:
  provider: "openai"         # "openai" | "hf_local"
  model: "gpt-3.5-turbo"     # or "llama-3.2-3b-instruct", "phi-2", etc.
  max_cache: 100000          # memoization for cost/latency
  round_decimals: 2          # round inputs before caching

active_learning:
  enabled: true
  query_per_episode: 32
  label_propagation_k: 128


```


