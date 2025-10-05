# Challenger-Based Combinatorial Bandits for Subcarrier Selection in OFDM Systems

This repository contains the reference implementation for **“Challenger‑Based Combinatorial Bandits for Subcarrier Selection in OFDM Systems.”**  
It frames OFDM subcarrier selection as a **Top‑m identification** problem and solves it using a challenger‑based confidence‑index method (exposed as `bandit: "CCS"`).

The code also includes classic synthetic settings (linear/logistic/etc.), but defaults to **OFDM** with realistic SNR modeling and estimation noise in **dB**.

---

## 📦 Repository structure

```
.
├── args_L.json                         # Large OFDM preset
├── args_M.json                         # Medium OFDM preset
├── args_S.json                         # Small/quick OFDM preset
├── main.py                             # Entry point / CLI runner
├── bandits.py                          # Bandit implementations incl. CCS
├── indices.py                          # Confidence indices (paired/disjoint, contextual & non-contextual)
├── betas.py                            # Exploration-rate (“beta”) schedules
├── learners.py                         # Online learners (FTL, AdaHedge, Fixed-Share)
├── data.py                             # Data builders, incl. OFDM channels/SNR/features
├── problems.py                         # Reward oracles; includes OFDM problem
├── utils.py                            # Plotting, LaTeX helpers
├── compare_complexity_constants.py     # LinGapE vs UGapE complexity utilities (linear case)
├── constants.py                        # Paths for Data/Results
└── requirements.txt                # Pinned dependencies
```

> **Paths:** results are written to `../Results/`, generated instances to `../Data/`, and DR data (if used) to `../DR_Data/` (see `constants.py`). Create those folders if you run from a different working directory.

---

## 🛰️ Problem & method (brief)

- **OFDM objective.** For a subset \(S\) of size \(m\):  
  \[
  r(S) = \sum_{i\in S}\log_2\!\big(1+\mathrm{SNR}_i\big)
  \]
- **Observation model.** True per‑subcarrier SNR is perturbed by **Gaussian noise in dB**, i.e., observations correspond to a log‑normal multiplicative error on linear SNR.
- **CCS (Challenger‑Based) algorithm.** Each round:
  1. Maintain a current best‑guess subset of size `m` and draw a **pool of challengers** (`n_das`).
  2. Use paired/disjoint contextual confidence indices \(B_{i,j}(t)\) to target comparisons.
  3. Sample to reduce uncertainty until a stopping rule certifies the Top‑m set (or a fixed budget ends).

Implementation uses indices from `indices.py`, rate schedules from `betas.py`, and the general experiment harness in `main.py`/`bandits.py`.

---

## 🔧 Installation

> Dependencies are pinned to legacy versions for reproducibility. A **Python 3.7** environment is recommended.

### Option A — Conda (recommended)
```bash
conda create -n ofdm-ccs python=3.7 -y
conda activate ofdm-ccs
pip install -r "requirements (2).txt"
```

### Option B — Virtualenv
```bash
python3.7 -m venv .venv
source .venv/bin/activate
pip install -r "requirements (2).txt"
```

**Notes**
- For `--mode generate_latex`, ensure a LaTeX toolchain (e.g., `texlive-full`) is installed.
- On Apple Silicon or very new Linux distros, Conda typically provides prebuilt wheels for the older SciPy/NumPy stack.

---

## ▶️ Quick start (OFDM)

From the repository root (or the directory containing `main.py`):

```bash
# Small/fast sanity check
python main.py --json_file args_S.json

# Medium
python main.py --json_file args_M.json

# Large (paper-scale)
python main.py --json_file args_L.json
```

Each JSON preset contains a full configuration. You can override any field on the CLI, e.g.:
```bash
python main.py --json_file args_M.json --n_das 50 --n_simu 200 --m 16 --sigma 0.5 --plot 1
```

Common flags (subset):
- `--json_file PATH` : load a configuration file (defaults to a preset in most workflows).
- `--mode {recommendation, small_test, finetuning, clear, generate_latex}`
- `--n_simu INT` : number of interaction rounds.
- `--plot {0,1}` / `--plot_rounds INT` : plotting controls.

Data / problem:
- `--data ofdm` and `--problem ofdm` (default in the OFDM presets).
- `--small_K INT` : number of subcarriers (arms) considered.
- `--small_N INT` : feature dimension.
- `--sigma FLOAT` : **dB** std‑dev for SNR estimation noise.

Bandit / algorithm:
- `--bandit CCS` : challenger‑based method.
- `--beta Heuristic` (or other schedules from `betas.py`).
- `--m INT` : number of subcarriers to select.
- `--n_das INT` : number of challengers drawn per round.
- `--epsilon FLOAT`, `--delta FLOAT` : accuracy/confidence parameters (if used by a stopping rule).
- `--use_chernoff {none, gaussian, bernouilli}` : optional stopping tweak.

Run `python main.py -h` for the full list.

---

## 📁 Outputs & artifacts

By default, results are written under `../Results/` as a folder whose name encodes dataset/problem and key parameters. Typical contents:

```
../Results/<run-name>/
├── parameters.json
├── scores_*.csv                # per-arm oracle scores used internally
├── features_*.csv              # feature matrix (e.g., N × K)
├── <plots>.png                 # learning curves / comparisons
└── ...
```

To **assemble a PDF** with plots for the current experiment:
```bash
python main.py --json_file args_L.json --mode generate_latex
```
(Requires LaTeX.)

To **clear** results:
```bash
python main.py --mode clear
```

---

## ⚙️ Configuration files

The three presets `args_S.json`, `args_M.json`, `args_L.json` define OFDM experiments of increasing scale. Typical keys include:
- `data`, `problem` — choose `"ofdm"` for OFDM experiments (other options exist in `data.py`).
- `bandit` — set to `"CCS"` for the challenger‑based algorithm.
- `beta` — exploration‑rate schedule, e.g. `"Heuristic"`.
- `m` — number of subcarriers to select.
- `sigma` — **dB** std‑dev for SNR estimation noise.
- `n_simu` — number of rounds.
- `small_K`, `small_N` — arms (subcarriers) and feature dimension.
- `n_das` — number of challengers per round.
- `epsilon`, `delta` — (optional) accuracy/confidence constants.
- `plot`, `plot_rounds`, `verbose` — visualization and logging knobs.

---

## 📡 OFDM modeling details (implemented)

- **Channels.** Rayleigh (default) and Rician options in `data.py`.
- **SNR.** \(\gamma_i = \frac{|h_i|^2 P}{\sigma^2}\), with configurable `P`/`Ptot`/`sigma2`.
- **Estimation noise.** Additive Gaussian noise in **dB** → multiplicative log‑normal on linear SNR.
- **Reward.** \(\log_2(1+\widehat{\mathrm{SNR}}_i)\) summed over the selected `m` subcarriers.
- **Features.** `data.py` builds an \(N \times K\) feature matrix used by contextual indices.

---

## 🧪 Reproducibility

- Random seeds are set at startup; each run writes a `parameters.json` snapshot inside its result folder.
- Matplotlib uses the non‑interactive **Agg** backend for headless plotting.

---

## ❓ Troubleshooting

- **Build errors for NumPy/SciPy on modern systems** → use the Conda setup above.
- **LaTeX errors** → install a TeX distribution or skip `--mode generate_latex`.
- **No output/plots** → verify `../Results/` exists and `matplotlib` is installed; set `--plot 1` if needed.
- **Paths look odd** → remember outputs are relative to script location (see `constants.py`).

---

## 📝 Citation

If you use this code, please cite:

> **Challenger‑Based Combinatorial Bandits for Subcarrier Selection in OFDM Systems**  
> Mohsen Amiri, *et al.* (Year).  
> *(Add venue / arXiv / DOI when available.)*

```bibtex
@inproceedings{amiri202X_ofdm_ccs,
  title     = {Challenger-Based Combinatorial Bandits for Subcarrier Selection in OFDM Systems},
  author    = {Amiri, Mohsen and ...},
  booktitle = {...},
  year      = {202X}
}
```

---

## 📄 License

Add a license (e.g., MIT) in `LICENSE`. If omitted, the default is “all rights reserved.”

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue to discuss what you’d like to change.

---

## ✉️ Contact

Maintainer: **Mohsen Amiri**  
E-mail: mohsen.amiri@dsv.su.se
