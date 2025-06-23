# QICGen – Inverse Quantum‑Circuit Synthesis via Conditional Generative Models

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue"/>
  <img src="https://img.shields.io/badge/license-MIT-green"/>
</p>

QICGen is a code‑base that trains **conditional generative models** (Transformers, discrete Diffusion models and Graph VAEs) to *invert* measurement statistics: given a desired probability distribution over computational‑basis outcomes, the models synthesise a quantum circuit whose measurement matches that distribution.

---

## 🗂 Repository layout

```
.
├── diffusion.py            # DDPM training / testing script
├── transformer.py          # Transformer training / testing script
├── graph_vae.py            # GraphVAE training / testing script
├── rnn.py                  # Lightweight GRU baseline (optional)
├── generate_dataset.py     # Create the 6 000‑sample stratified dataset
├── entropy.py              # Quick entropy histogram helper
├── plotting.py             # Post‑hoc figure generation (val‑loss curves, etc.)
├── visualize_circuits.py   # Inspect random dataset samples
└── data/                   # Auto‑generated JSON datasets live here
```

---

## ⚡ Quick start

```bash
# 1. Clone repository
$ git clone https://github.com/Girish-Krishnan/Inverse-QC-Design.git
$ cd Inverse-QC-Design

# 2. Create environment (conda example)
$ conda create -n qicgen python=3.9 -y
$ conda activate qicgen

# 3. Install Python dependencies
$ pip install -r requirements.txt     # provided in the repo root

# (Optional) Verify GPU availability
$ python -c "import torch, sys; print(torch.cuda.is_available())"
```

### Requirements

| Package                                      | Tested version |
| -------------------------------------------- | -------------- |
| `torch`                                      |  2.1.0         |
| `qiskit`                                     |  0.45.0        |
| `matplotlib`                                 |  3.8           |
| `tqdm`                                       | 4.66           |
| `networkx`, `numpy`, `scipy`, `scikit‑learn` | recent         |

> **GPU** All models run on CPU, but Diffusion & GraphVAE *strongly* benefit from CUDA‑enabled GPUs (≥ 8 GB recommended).

---

## 📊 Dataset generation

The dataset contains 6 000 circuits, stratified into six Shannon‑entropy bins across 2–6 qubits.

```bash
$ python generate_dataset.py
```

This creates three files in `data/`:

```
data/
 ├── crypto_qc_train.json   # 80 % (4 800 samples)
 ├── crypto_qc_val.json     # 10 % (   600 samples)
 └── crypto_qc_test.json    # 10 % (   600 samples)
```

Each entry is a JSON dict containing:

```jsonc
{
  "structure": "GHZ",         // family
  "num_qubits": 4,
  "tokens": ["h q0", "cx q0 q1", ...],
  "probs": {"0000": 0.5, "1111": 0.5, ...},
  "entropy": 1.000,
  "is_entangled": true
}
```

Use `visualize_circuits.py` to sanity‑check random samples:

```bash
$ python visualize_circuits.py --file data/crypto_qc_train.json --n 3
```

---

## 🏋️‍♂️ Training models

Below we show minimal commands that reproduce the default hyper‑parameters from the paper. All checkpoints, logs and visualizations are written to `results/<model>/`.

### 1. Transformer

```bash
$ python transformer.py train \
     --train_json data/crypto_qc_train.json \
     --val_json   data/crypto_qc_val.json   \
     --epochs 100 --batch_size 32 --lr 1e-3 \
     --output_dir results/transformer
```

### 2. Discrete Diffusion (DDPM)

```bash
$ python diffusion.py train \
     --train_json data/crypto_qc_train.json \
     --val_json   data/crypto_qc_val.json   \
     --epochs 100 --batch_size 64 --lr 2e-4 \
     --hidden_dim 256 --T 200 \
     --output_dir results/diffusion
```

### 3. Graph VAE

```bash
$ python graph_vae.py train \
     --train_json data/crypto_qc_train.json \
     --val_json   data/crypto_qc_val.json   \
     --epochs 100 --batch_size 32 --lr 1e-3 \
     --hidden_dim 256 --latent_dim 128 \
     --output_dir results/graphvae
```

> **Tip** Set `--device cuda` on any command to enable GPU training.

A light‑weight GRU baseline is available:

```bash
$ python rnn.py --mode train --train_json data/crypto_qc_train.json --val_json data/crypto_qc_val.json
```

---

## 🔍 Monitoring training

During training each script:

* logs metrics to a `training_logs.json`,
* dumps best checkpoint to `checkpoints/`,
* saves periodic qualitative samples to `visuals/`.

Use `plotting.py` to aggregate the three models:

```bash
$ python plotting.py   # outputs .png figures to repo root
```

---

## 🧪 Testing & circuit search

Each model provides a `test` sub‑command that, given a target distribution, searches for **exact‑match** circuits (default = 5). Results are visualized as a bar chart plus circuit diagrams.

```bash
$ python diffusion.py test \
     --test_json data/crypto_qc_test.json \
     --model_path results/diffusion/checkpoints/ddpm_best.pt \
     --output_vis diffusion_vis.png
```

Equivalent commands exist for `transformer.py` and `graph_vae.py`.

Parameters of interest:

* `--target_index` – choose specific test entry (default random).
* `--num_matches` – number of distinct circuits to retrieve.
* `--max_attempts` – sampling budget before giving up.

---

## 🛡️ License

This project is distributed under the **MIT License**.  See `LICENSE` for details.
