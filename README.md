# Reward-Shaping-LLM
**LLM-Enhanced Reinforcement Learning for Time Series Anomaly Detection**  
LSTM-DQN agent + LLM-derived semantic potentials + VAE reconstruction guidance + active learning/label propagation.

[📄 Paper (PDF)](./paper/IEEE_Conference__LLM_Based_Potential_Reward.pdf) 

<div align="center">
  <img
    src="Figure/Proposed-LLM.png"
    width="600"
    height="250"
    alt="Proposed Figure">
</div>

---

## ✨ What’s inside
- **LLM-based Potential-Based Reward Shaping (PBRS)**  
  Policy-invariant shaping with a potential $\phi(s)$ predicted by an LLM over sliding windows:  
  $r'(s,a,s') = r(s,a,s') + \gamma\,\phi(s') - \phi(s)$
- **Dynamic reward blending**  
  Mix supervised classification feedback with VAE reconstruction error using a time-varying controller $\lambda(t)$.
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


```
## 🔧 Configuration
```bash
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


## 🧱 Reward construction

- **Base classification reward:** TP = +5, TN = +1, FP = −1, FN = −5

- **VAE augmentation (R₂):** add reconstruction MSE

- **Dynamic mixing:**

$$
R_{\text{total}} = R_1 + \lambda(t)\,R_2
$$

$$
\lambda_{t+1} = \operatorname{clip}\!\big(\lambda_t + \alpha\,[R_{\text{target}} - R_{\text{episode}}],\, \lambda_{\min},\, \lambda_{\max}\big)
$$

- **PBRS with LLM:**

$$
r' = r + \gamma\,\phi(s') - \phi(s)
$$


## ▶️ Quick start
```bash

# Train on Yahoo-A1 with Llama-3.2-3B potential shaping
python train.py \
  --config configs/base.yaml \
  --data.name yahoo_a1 \
  --llm.provider hf_local --llm.model llama-3.2-3b-instruct

# Evaluate a trained checkpoint
python evaluate.py \
  --checkpoint runs/yahoo_a1_llama3/best.ckpt \
  --metrics precision recall f1


```


## 🔍 Active learning

After each episode, compute the **margin** between the top two actions’ Q-values:

$$
\mathrm{Margin}(s) = \big|\,Q(s, a_0) - Q(s, a_1)\,\big|
$$

Query the lowest-margin samples and propagate labels to their nearest neighbors (e.g., k-NN in an embedding space).



## 📊 Example target results

*(Exact values depend on seeds and implementation details.)*

| Dataset  | LLM          | Precision | Recall |   F1   |
|----------|--------------|----------:|------:|------:|
| Yahoo-A1 | Llama-3.2-3B |    0.6051 | 0.9565 | 0.7413 |
| SMD      | Llama-3.2-3B |    0.3813 | 0.8685 | 0.5300 |

**Interpretation.** Llama-3 variants often balance precision/recall well; GPT-3.5 may push recall higher on Yahoo but can reduce precision; smaller models (e.g., Phi-2) can be conservative on SMD.



## 📣 Citation

If you use this code or ideas, please cite:
```
@inproceedings{golchin2025llm_reward_shaping,
  title     = {LLM-Enhanced Reinforcement Learning for Time Series Anomaly Detection},
  author    = {Golchin, Bahareh and Rekabdar, Banafsheh and Justo, Danielle},
  booktitle = {IEEE 20th International Conference on Semantic Computing (ICSC)},
  year      = {2025}
}
```
