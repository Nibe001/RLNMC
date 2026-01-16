# RLNMC — Reinforcement Learning Nonlocal Monte Carlo for k-SAT

This repository provides a **modular, and reproducible implementation** of three heuristic solvers for hard combinatorial optimization problems, with a focus on **4-SAT / MAX-4-SAT**:

- **Simulated Annealing (SA)**
- **Nonlocal Monte Carlo (NMC)**
- **Reinforcement Learning Nonlocal Monte Carlo (RLNMC)**

The project reproduces, at a qualitative and methodological level, the main ideas and experimental protocol of:

> **Dobrynin et al., 2025 — “Nonlocal Monte Carlo via Reinforcement Learning”**

while deliberately using a **simpler and fully PyTorch-based stack** (stable-baselines3 PPO + MLP policy).

---

## 1. Overview

Hard k-SAT instances exhibit **rugged energy landscapes** with:
- many local minima,
- rigid (frozen) variables,
- large energy barriers.

Standard local MCMC methods such as Simulated Annealing often **freeze prematurely**.  
NMC alleviates this by introducing **nonlocal moves** that jointly excite rigid variables (the *backbone*).

RLNMC goes one step further by **learning how to select backbone variables** using reinforcement learning, instead of relying on fixed heuristics.

This repository implements:
- the algorithms,
- instance generators (uniform and scale-free),
- a Gym-compatible RL environment,
- training with PPO,
- evaluation metrics (residual energy, TTS, diversity),
- automatic plotting and analysis.

---

## 2. Project Structure

```
RLNMC-project/
├── algorithms/
│   ├── simulated_annealing.py    # SA
│   ├── nonlocal_mc.py            # NMC
│   └── rl_nmc.py                 # RLNMC (inference wrapper)
│
├── env/
│   └── sat_env.py                # SATProblem + RLNMC Gym environment
│
├── data/
│   ├── generate_instances.py     # Uniform & scale-free 4-SAT generation
│   └── instances/               # Saved instances (JSON / DIMACS)
│
├── metrics/
│   └── metrics.py                # Residual energy, TTS_99, diversity
│
├── plots/
│   └── plotting.py               # Matplotlib plotting utilities
│
├── training/
│   ├── train_policy.py           # PPO training for RLNMC
│   ├── run_experiments.py        # Benchmark runner
│   └── analyze_results.py        # Metrics + figures
│
├── notebooks/
│   ├── RLNMC_Experiments.ipynb   # Interactive experiments
│   └── RLNMC_Analysis.ipynb      # Optional deeper analysis
│
├── configs/
│   └── config.yaml               # Centralized configuration
│
├── requirements.txt
└── README.md
```

---

## 3. Installation

### 3.1 Create environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch
- stable-baselines3
- gymnasium
- NumPy / SciPy
- matplotlib
- networkx
- PyYAML

No proprietary solvers are required.

---

## 4. Instance Generation

### Uniform random 4-SAT
```bash
python -m data.generate_instances \
  --type uniform \
  --n_vars 250 \
  --clause_ratio 9.2 \
  --n_instances 50 \
  --k 4 \
  --planted \
  --out data/instances/uniform.json
```

### Scale-free 4-SAT (industrial-like)
```bash
python -m data.generate_instances \
  --type scalefree \
  --n_vars 250 \
  --clause_ratio 9.2 \
  --n_instances 50 \
  --k 4 \
  --alpha 2.6 \
  --planted \
  --out data/instances/scalefree.json
```

- Scale-free instances follow a **power-law variable frequency** (Ansótegui et al., IJCAI 2009).
- Planted mode guarantees a solution of energy 0 (useful for TTS).

---

## 5. Training the RLNMC Policy

RLNMC is trained using **PPO with an MLP policy**.

Example (scale-free 4-SAT, N=250):

```bash
python -m training.train_policy \
  --problem_type scalefree \
  --n_vars 250 \
  --k 4 \
  --clause_ratio 9.2 \
  --alpha 2.6 \
  --planted \
  --total_timesteps 200000 \
  --n_envs 8 \
  --seed 123 \
  --save_dir training/checkpoints/sf250
```

Outputs:
- PPO checkpoints (`ppo_rlnmc_*.zip`)
- final model (`ppo_rlnmc_final.zip`)
- TensorBoard logs

The reward is defined as **improvement of the best energy found so far**, following the paper.

---

## 6. Running Experiments (SA / NMC / RLNMC)

### Example benchmark run

```bash
python -m training.run_experiments \
  --instances_path data/instances/scalefree.json \
  --algos sa nmc rlnmc \
  --rlnmc_model training/checkpoints/sf250/ppo_rlnmc_final.zip \
  --budgets 200 500 1000 2000 \
  --replicas 50 \
  --out results/scalefree_results.json
```

This produces a single JSON file containing:
- per-run results,
- energy traces,
- solutions (for diversity),
- full metadata.

---

## 7. Analysis and Figures

```bash
python -m training.analyze_results \
  --input results/scalefree_results.json \
  --out figures/
```

Automatically generates:
- **Residual energy vs sweeps**
- **TTS₉₉ vs sweeps**
- **Diversity curves D(R)**
- **Diversity integral summary**

Figures are saved in `figures/`.

---

## 8. Metrics Implemented

### Residual Energy
Median (or mean) best-so-far energy above the optimum as a function of compute budget.

### Time-To-Solution (TTS_99)
Estimated time required to reach 99% success probability:
```
$$
\mathrm{TTS}_{99} = t \cdot \frac{\log(0.01)}{\log(1 - p_{\text{succ}})}
$$
```

with confidence intervals (Wilson).

### Diversity
- Pairwise Hamming distances between solutions
- Construction of a proximity graph
- Greedy approximation of Maximum Independent Set
- Diversity integral (area under D(R))

---

## 9. Design Choices and Differences with the Original Paper

| Aspect | This repository | Original RLNMC |
|------|-----------------|----------------|
| Framework | PyTorch + SB3 | JAX / Flax |
| Policy | MLP (feedforward) | GNN + GRU |
| Backbone selection | Learned (Bernoulli mask) | Learned |
| MIS for diversity | Greedy approximation | Exact ILP |
| Parallelism | VecEnv (SB3) | Massive JAX vectorization |

The goal here is **clarity, reproducibility, and pedagogical value**, not raw performance parity.

---

## 10. Extensions and Future Work

Possible improvements:
- Replace MLP with **Graph Neural Network**
- Add **recurrent policy** (LSTM/GRU)
- Curriculum learning on instance size
- Alternative reward shaping
- Test on **industrial SAT benchmarks**
- Generalize to other problems (Max-Cut, Ising, etc.)

---

## 11. References

- Dobrynin et al., *Nonlocal Monte Carlo via Reinforcement Learning*, 2025
- Mohseni et al., *Nonequilibrium Nonlocal Monte Carlo*, 2023
- Ansótegui et al., *Structure of Industrial SAT Instances*, IJCAI 2009
- Friedrich et al., *Phase Transitions in Scale-Free SAT*, AAAI 2017

---

## 12. License

This project is provided for **academic and educational use**.
