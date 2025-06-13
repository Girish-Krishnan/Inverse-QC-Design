#!/usr/bin/env python3
"""
train_diffusion_inverse.py

Train or test a discrete denoising–diffusion model (DDPM) conditioned on
target measurement distributions, for inverse quantum circuit synthesis.

Usage (training):
  python train_diffusion_inverse.py train \
      --train_json    data/crypto_qc_train.json \
      --val_json      data/crypto_qc_val.json \
      --epochs        30 \
      --batch_size    64 \
      --lr            2e-4 \
      --hidden_dim    256 \
      --latent_dim    128 \
      --max_len       32 \
      --T             200 \
      --save_interval 5 \
      --n_samples     3 \
      --output_dir    diffusion_results \
      --device        cuda

Usage (testing):
  python train_diffusion_inverse.py test \
      --test_json    data/crypto_qc_test.json \
      --model_path   diffusion_results/checkpoints/ddpm_best.pt \
      --hidden_dim   256 \
      --latent_dim   128 \
      --max_len      32 \
      --T            200 \
      --target_index 10 \
      --num_matches  5 \
      --max_attempts 5000 \
      --output_vis   diffusion_test_vis.png \
      --device       cuda
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix

# ------------------------------------------------------------
# Tokenizer (identical vocabulary to other models)
# ------------------------------------------------------------
class Tokenizer:
    """
    Maps tokens (gate strings) to integer IDs, and vice versa.
    Must match the dataset's token vocabulary exactly.
    """
    def __init__(self, max_qubits: int = 6):
        base_tokens = ["PAD", "START", "END"]
        gates = ["h", "x", "y", "z", "cx", "rx", "ry", "rz"]
        tokens = base_tokens[:]
        for g in gates:
            for q1 in range(max_qubits):
                if g == "cx":
                    for q2 in range(max_qubits):
                        if q1 != q2:
                            tokens.append(f"{g} q{q1} q{q2}")
                elif g in ["rx", "ry", "rz"]:
                    for angle in range(8):
                        tokens.append(f"{g}({angle}) q{q1}")
                else:
                    tokens.append(f"{g} q{q1}")
        tokens += [f"bb84_step q{i}" for i in range(max_qubits)]
        tokens += [f"e91_entangle q{i} q{j}"
                   for i in range(max_qubits) for j in range(max_qubits) if i != j]
        tokens += ["ghz_step q0"] + [f"ghz_entangle q0 q{i}" for i in range(1, max_qubits)]
        self.vocab = {tok: idx for idx, tok in enumerate(tokens)}
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def encode(self, seq, max_len: int = 32):
        """
        Encode a sequence of token strings into a list of IDs of length max_len.
        Pads or truncates to exactly max_len.
        """
        ids = [self.vocab.get(t, self.vocab["PAD"]) for t in seq][: max_len - 2]
        padded = [self.vocab["START"], *ids, self.vocab["END"]]
        if len(padded) < max_len:
            padded += [self.vocab["PAD"]] * (max_len - len(padded))
        return padded

    def decode(self, ids):
        """
        Decode a list of IDs into token strings, skipping PAD/START/END.
        """
        special = {self.vocab["PAD"], self.vocab["START"], self.vocab["END"]}
        return [self.inv_vocab[i] for i in ids if i not in special]

    def __len__(self):
        return len(self.vocab)

# ------------------------------------------------------------
# QuantumDataset + custom collate_fn
# ------------------------------------------------------------
class QuantumDataset(Dataset):
    """
    PyTorch Dataset for diffusion model:
      returns (seq_ids, p_target, num_qubits)
    """
    def __init__(self, json_path: str, tokenizer: Tokenizer, max_len: int = 32):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_states = 2 ** 6  # always pad distributions to length 64

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        tokens = entry["tokens"]
        nq = entry["num_qubits"]
        raw_probs = entry["probs"]

        # Encode token sequence to fixed length
        seq_ids = self.tokenizer.encode(tokens, self.max_len)
        seq_ids = torch.tensor(seq_ids, dtype=torch.long)

        # Build p_target (length 2^nq, then pad to 64)
        p = np.zeros(2 ** nq, dtype=np.float32)
        for i_state in range(2 ** nq):
            bitstr = format(i_state, f"0{nq}b")
            p[i_state] = raw_probs.get(bitstr, 0.0)
        if p.shape[0] < self.max_states:
            p = np.pad(p, (0, self.max_states - p.shape[0]))
        p_target = torch.tensor(p, dtype=torch.float32)

        return seq_ids, p_target, nq

def collate_fn(batch):
    seq_ids, p_targets, nqs = zip(*batch)
    seq_ids = torch.stack(seq_ids, dim=0)
    p_targets = torch.stack(p_targets, dim=0)
    nqs = torch.tensor(nqs, dtype=torch.long)
    return seq_ids, p_targets, nqs

# ------------------------------------------------------------
# Discrete Diffusion Utilities
# ------------------------------------------------------------
class DiscreteDiffusion:
    """
    Forward q(x_t | x_{t-1}) with token‐wise corruption.
    """
    def __init__(self, vocab_size: int, T: int = 200,
                 beta_start: float = 1e-4, beta_end: float = 5e-2, device="cpu"):
        self.vocab = vocab_size
        self.T = T
        self.device = device
        betas = torch.linspace(beta_start, beta_end, T, device=device)  # (T,)
        self.betas = betas
        self.keep_probs = 1.0 - betas       # p_keep at each timestep
        # cumulative product for q(x_t | x_0)
        self.cumprod = torch.cumprod(self.keep_probs, dim=0)            # ᾱ_t

    def sample_timesteps(self, B):
        """
        Sample uniform timesteps t in [0, T-1] for a batch of size B.
        """
        return torch.randint(0, self.T, (B,), device=self.device)

    def q_sample(self, x0: torch.LongTensor, t: torch.LongTensor):
        """
        Produce corrupted x_t given original x0 at timestep t.
        x0: (B, L) token IDs
        t:  (B,)  integer timesteps
        """
        B, L = x0.shape
        keep_prob = self.keep_probs[t].unsqueeze(1)  # (B,1)
        mask = torch.bernoulli(keep_prob).bool()     # (B, L) True→keep original
        rand_tokens = torch.randint(0, self.vocab, (B, L), device=x0.device)
        return torch.where(mask, x0, rand_tokens)

# ------------------------------------------------------------
# Diffusion Model: Transformer Denoiser
# ------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.LongTensor):
        half = self.dim // 2
        freqs = torch.exp(-np.log(self.max_period) * torch.arange(half, device=t.device) / half)
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, emb[:, :1]], dim=1)
        return emb  # (B, dim)

class DiffusionTransformer(nn.Module):
    def __init__(self, vocab_size, p_dim=64, max_len=32,
                 hidden=256, nhead=8, layers=6):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.p_embed = nn.Linear(p_dim + 1, hidden)       # +1 for normalized num_qubits
        self.time_enc = SinusoidalTimeEmbedding(hidden)

        self.transformer = nn.Transformer(
            d_model=hidden,
            nhead=nhead,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            batch_first=True
        )
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x_t, p_target, nq, t):
        """
        x_t      : (B, L) corrupted token IDs
        p_target : (B, 64) probability vector
        nq       : (B,) number of qubits
        t        : (B,) integer timestep
        returns: logits over vocabulary: (B, L, V)
        """
        B, L = x_t.shape
        dev = x_t.device

        # Embed tokens + positional
        pos = torch.arange(L, device=dev).unsqueeze(0)  # (1, L)
        tok_embed = self.tok_emb(x_t) + self.pos_emb(pos)  # (B, L, hidden)

        # Time embedding
        t_emb = self.time_enc(t)  # (B, hidden)
        tok_embed = tok_embed + t_emb.unsqueeze(1)  # (B, L, hidden)

        # Conditioning: p_target + normalized nq
        nq_norm = nq.float().unsqueeze(1) / 6.0  # (B,1)
        cond = torch.cat([p_target, nq_norm], dim=1)  # (B, 65)
        enc_ctx = self.p_embed(cond).unsqueeze(1)  # (B,1,hidden)
        enc_ctx = enc_ctx.repeat(1, L, 1)          # (B,L,hidden)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(dev)
        out = self.transformer(enc_ctx, tok_embed, tgt_mask=tgt_mask)  # (B, L, hidden)
        return self.out(out)  # (B, L, vocab_size)

# ------------------------------------------------------------
# Training / Validation Loss Computation
# ------------------------------------------------------------
def kl_divergence(p, q):
    """
    KL(p || q) for discrete distributions p,q (as float tensors).
    p>0 where applicable.
    """
    mask = p > 0
    return float((p[mask] * torch.log(p[mask] / (q[mask] + 1e-10))).sum().item())

def simulate_qc(tokens, nq):
    """
    Build a Qiskit QuantumCircuit from token strings; return probabilities_dict.
    """
    qc = QuantumCircuit(nq)
    for tok in tokens:
        parts = tok.split()
        op = parts[0]
        if op in {"h","x","y","z"} and len(parts) == 2:
            q = int(parts[1][1:])
            if q < nq:
                getattr(qc, op)(q)
        elif op == "cx" and len(parts) == 3:
            q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
            if q1 < nq and q2 < nq:
                qc.cx(q1, q2)
        elif op in {"rx","ry","rz"} and len(parts) == 2:
            gate_name = op[:2]
            bucket = int(op[3:-1])
            angle = bucket * 2 * np.pi / 8
            q = int(parts[1][1:])
            if q < nq:
                getattr(qc, gate_name)(angle, q)
        elif op in {"e91_entangle","ghz_entangle"} and len(parts) == 3:
            q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
            if q1 < nq and q2 < nq:
                qc.h(q1); qc.cx(q1, q2)
        elif op == "ghz_step" and len(parts) == 2:
            q0 = int(parts[1][1:])
            if q0 < nq:
                qc.h(q0)
    sv = Statevector.from_instruction(qc)
    return sv.probabilities_dict()

def entanglement_entropy_loss(true_probs, pred_probs, num_qubits):
    """
    Compute |S_true - S_pred| where S = -Tr( rho_sub log rho_sub ).
    Uses Qiskit to reconstruct statevectors from probability vectors.
    """
    try:
        n = num_qubits.item()
        sv_true = Statevector.from_probabilities(true_probs[: 2**n].cpu().numpy())
        sv_pred = Statevector.from_probabilities(pred_probs[: 2**n].cpu().numpy())
        rho_t = DensityMatrix(partial_trace(sv_true, list(range(1, n))))
        rho_p = DensityMatrix(partial_trace(sv_pred, list(range(1, n))))
        s_t = rho_t.entropy()
        s_p = rho_p.entropy()
        return torch.tensor(abs(s_t - s_p), device=true_probs.device, dtype=torch.float32)
    except Exception:
        return torch.tensor(0.0, device=true_probs.device, dtype=torch.float32)

# ------------------------------------------------------------
# Visualization Utility during Training
# ------------------------------------------------------------
def save_sample_vis(model, dataset, tokenizer, epoch, out_dir, device, n_samples=3):
    """
    Pick n_samples random validation entries:
      - Corrupt x0 → run model denoiser chain to reconstruct x0_hat
      - Decode x0_hat to tokens, simulate, compare to ground truth p_target
      - Plot reconstructed circuit diagram + bar chart GT vs generated
    """
    sos_id = tokenizer.vocab["START"]
    eos_id = tokenizer.vocab["END"]
    chosen = random.sample(range(len(dataset)), n_samples)

    for i, idx in enumerate(chosen, start=1):
        x0_ids, p_target, nq = dataset[idx]
        x0_ids = x0_ids.to(device).unsqueeze(0)        # (1, L)
        p_target = p_target.to(device).unsqueeze(0)    # (1, 64)
        nq_tensor = torch.tensor([nq], device=device)  # (1,)
        L = x0_ids.shape[1]

        # Run through full diffusion chain: sample t=T-1, then denoise to t=0 (greedy)
        x_t = x0_ids.clone()
        with torch.no_grad():
            for t_step in reversed(range(diffusion.T)):
                t_b = torch.tensor([t_step], device=device)
                logits = model(x_t, p_target, nq_tensor, t_b)  # (1, L, V)
                x_pred = logits.argmax(dim=-1)                # (1, L)
                if t_step > 0:
                    x_t = x_pred  # feed predicted back as x_t for next
            x0_hat = x_pred.squeeze(0).cpu().tolist()  # predicted sequence IDs

        gen_tokens = tokenizer.decode(x0_hat)

        # Simulate generated circuit
        try:
            pred_dict = simulate_qc(gen_tokens, nq)
        except:
            pred_dict = {}

        # Ground-truth dictionary
        raw_probs = {format(i, f"0{nq}b"): float(p_target[0, i].cpu().item())
                     for i in range(2**nq)}

        states = [format(x, f"0{nq}b") for x in range(2**nq)]
        gt_vals = [raw_probs.get(s, 0.0) for s in states]
        pr_vals = [pred_dict.get(s, 0.0) for s in states]

        fig, (ax_circ, ax_bar) = plt.subplots(1, 2, figsize=(10, 4),
                                              gridspec_kw={"width_ratios": [1, 2]})
        qc = QuantumCircuit(nq)
        for tok in gen_tokens:
            parts = tok.split()
            op = parts[0]
            if op in {"h","x","y","z"} and len(parts) == 2:
                q = int(parts[1][1:])
                if q < nq:
                    getattr(qc, op)(q)
            elif op == "cx" and len(parts) == 3:
                q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
                if q1 < nq and q2 < nq:
                    qc.cx(q1, q2)
            elif op in {"rx","ry","rz"} and len(parts) == 2:
                gate_name = op[:2]
                bucket = int(op[3:-1])
                angle = bucket * 2 * np.pi / 8
                q = int(parts[1][1:])
                if q < nq:
                    getattr(qc, gate_name)(angle, q)
            elif op in {"e91_entangle","ghz_entangle"} and len(parts) == 3:
                q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
                if q1 < nq and q2 < nq:
                    qc.h(q1); qc.cx(q1, q2)
            elif op == "ghz_step" and len(parts) == 2:
                q0 = int(parts[1][1:])
                if q0 < nq:
                    qc.h(q0)
        try:
            qc.draw(output="mpl", ax=ax_circ)
        except:
            ax_circ.text(0.5, 0.5, "Circuit\nunavailable", ha="center", va="center")
        ax_circ.set_title(f"Epoch {epoch} Sample {i} ({nq} qubits)")
        ax_circ.axis("off")

        x = np.arange(len(states))
        ax_bar.bar(x - 0.15, gt_vals, width=0.3, label="Ground Truth", color="steelblue")
        ax_bar.bar(x + 0.15, pr_vals, width=0.3, label="Reconstructed", color="orange")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(states, rotation=90, fontsize=6)
        ax_bar.set_ylabel("Probability")
        ax_bar.set_title("Full Spectrum")
        ax_bar.legend()
        ax_bar.set_xlim(-0.5, len(states) - 0.5)

        plt.tight_layout()
        save_path = out_dir / f"epoch{epoch:03d}_sample{i}.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

# ------------------------------------------------------------
# Training routine
# ------------------------------------------------------------
def train_loop(args):
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    vis_dir = out_dir / "visuals"
    ckpt_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    tokenizer = Tokenizer(max_qubits=6)
    train_ds = QuantumDataset(args.train_json, tokenizer, max_len=args.max_len)
    val_ds = QuantumDataset(args.val_json, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_fn)

    global diffusion  # so save_sample_vis can access
    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu")

    diffusion = DiscreteDiffusion(len(tokenizer), T=args.T, device=device)
    model = DiffusionTransformer(len(tokenizer), p_dim=64, max_len=args.max_len,
                                 hidden=args.hidden_dim, nhead=8, layers=6).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["PAD"])

    logs = {"epoch": [], "train_CE": [], "val_CE": []}
    best_val_ce = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_ce = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for x0, p_target, nq in pbar:
            x0 = x0.to(device)               # (B, L)
            p_target = p_target.to(device)   # (B, 64)
            nq = nq.to(device)               # (B,)

            B, L = x0.shape
            t = diffusion.sample_timesteps(B)        # (B,)
            x_t = diffusion.q_sample(x0, t)          # corrupted tokens

            logits = model(x_t, p_target, nq, t)     # (B, L, V)
            loss = ce(logits.view(-1, logits.size(-1)), x0.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_ce += loss.item()
            pbar.set_postfix(train_CE=running_ce / ((pbar.n + 1) or 1))

        train_ce_epoch = running_ce / len(train_loader)
        logs["epoch"].append(epoch)
        logs["train_CE"].append(train_ce_epoch)

        # Validation
        model.eval()
        val_ce_accum = 0.0
        with torch.no_grad():
            for x0, p_target, nq in val_loader:
                x0 = x0.to(device)
                p_target = p_target.to(device)
                nq = nq.to(device)

                B, L = x0.shape
                t = diffusion.sample_timesteps(B)
                x_t = diffusion.q_sample(x0, t)

                logits = model(x_t, p_target, nq, t)
                val_ce_accum += ce(logits.view(-1, logits.size(-1)), x0.view(-1)).item()
        val_ce_epoch = val_ce_accum / len(val_loader)
        logs["val_CE"].append(val_ce_epoch)

        print(f"Epoch {epoch:03d} | Train CE={train_ce_epoch:.4f} | Val CE={val_ce_epoch:.4f}")

        # Save best checkpoint
        if val_ce_epoch < best_val_ce:
            best_val_ce = val_ce_epoch
            ckpt = {
                "model_state_dict": model.state_dict(),
                "diffusion_T": args.T,
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "max_len": args.max_len,
                "vocab_size": len(tokenizer),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch
            }
            ckpt_path = ckpt_dir / "ddpm_best.pt"
            torch.save(ckpt, ckpt_path)
            print(f"  → New best model saved: Val CE {best_val_ce:.4f}")

        # Periodic visualizations
        if epoch % args.save_interval == 0:
            epoch_vis_dir = vis_dir / f"epoch_{epoch:03d}"
            epoch_vis_dir.mkdir(exist_ok=True)
            save_sample_vis(model, val_ds, tokenizer, epoch, epoch_vis_dir, device, n_samples=args.n_samples)

        with open(out_dir / "training_logs.json", "w") as lf:
            json.dump(logs, lf, indent=2)

    print("Training complete.")

# ------------------------------------------------------------
# Testing routine
# ------------------------------------------------------------
def test_loop(args):
    """
    In test mode:
      - Load trained DDPM checkpoint
      - Load test JSON
      - Select a target distribution by index or random
      - Sample x0_hat by ancestral sampling (greedy) from noise
      - Repeat until num_matches exact matches found
      - Plot bar chart GT + subplots of matching circuits
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    tokenizer = Tokenizer(max_qubits=6)
    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)

    model = DiffusionTransformer(checkpoint["vocab_size"], p_dim=64, max_len=checkpoint["max_len"],
                                 hidden=checkpoint["hidden_dim"], nhead=8, layers=6).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = DiscreteDiffusion(checkpoint["vocab_size"], T=checkpoint["diffusion_T"], device=device)

    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    test_ds = []
    for entry in test_data:
        nq = entry["num_qubits"]
        raw_probs = entry["probs"]
        p = np.zeros(2 ** nq, dtype=np.float32)
        for i_state in range(2 ** nq):
            bitstr = format(i_state, f"0{nq}b")
            p[i_state] = raw_probs.get(bitstr, 0.0)
        if p.shape[0] < 64:
            p = np.pad(p, (0, 64 - p.shape[0]))
        test_ds.append((torch.tensor(p, dtype=torch.float32), nq, raw_probs))

    if args.target_index is not None and 0 <= args.target_index < len(test_ds):
        idx = args.target_index
    else:
        idx = random.randrange(len(test_ds))

    p_target_tensor, nq, gt_dict = test_ds[idx]
    p_target_tensor = p_target_tensor.to(device)

    print(f"Selected test index {idx} (n_qubits={nq}). Searching for {args.num_matches} exact matches...")

    matches = []
    attempts = 0
    seen_keys = set()
    full_states = [format(i, f"0{nq}b") for i in range(2 ** nq)]

    while len(matches) < args.num_matches and attempts < args.max_attempts:
        attempts += 1
        # Start from random noise
        x_t = torch.randint(0, len(tokenizer), (1, args.max_len), device=device)
        with torch.no_grad():
            # Denoise from t=T-1 down to t=0
            for t_step in reversed(range(diffusion.T)):
                t_b = torch.tensor([t_step], device=device)
                logits = model(x_t, p_target_tensor.unsqueeze(0), torch.tensor([nq], device=device), t_b)
                x_t = logits.argmax(dim=-1)  # (1, L)
            x0_hat_ids = x_t.squeeze(0).cpu().tolist()

        gen_tokens = tokenizer.decode(x0_hat_ids)
        key = "|".join(gen_tokens) + f"_n={nq}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        try:
            pred_dict = simulate_qc(gen_tokens, nq)
        except:
            continue

        is_match = True
        for state in full_states:
            if abs(gt_dict.get(state, 0.0) - pred_dict.get(state, 0.0)) > 1e-9:
                is_match = False
                break
        if is_match:
            matches.append(gen_tokens)

    print(f"  → Found {len(matches)} matches in {attempts} attempts.")

    fig = plt.figure(figsize=(4 + 3 * args.num_matches, 4))
    ax_bar = fig.add_subplot(1, args.num_matches + 1, 1)
    gt_vals = [gt_dict.get(s, 0.0) for s in full_states]
    x = np.arange(len(full_states))
    ax_bar.bar(x, gt_vals, color="steelblue")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(full_states, rotation=90, fontsize=6)
    ax_bar.set_ylabel("Probability")
    ax_bar.set_title(f"Target Distribution (n={nq})")
    ax_bar.set_xlim(-0.5, len(full_states) - 0.5)

    for i, gen_tokens in enumerate(matches, start=1):
        ax_circ = fig.add_subplot(1, args.num_matches + 1, i + 1)
        qc = QuantumCircuit(nq)
        for tok in gen_tokens:
            parts = tok.split()
            op = parts[0]
            if op in {"h","x","y","z"} and len(parts) == 2:
                q = int(parts[1][1:])
                if q < nq:
                    getattr(qc, op)(q)
            elif op == "cx" and len(parts) == 3:
                q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
                if q1 < nq and q2 < nq:
                    qc.cx(q1, q2)
            elif op in {"rx","ry","rz"} and len(parts) == 2:
                gate_name = op[:2]
                bucket = int(op[3:-1])
                angle = bucket * 2 * np.pi / 8
                q = int(parts[1][1:])
                if q < nq:
                    getattr(qc, gate_name)(angle, q)
            elif op in {"e91_entangle","ghz_entangle"} and len(parts) == 3:
                q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
                if q1 < nq and q2 < nq:
                    qc.h(q1); qc.cx(q1, q2)
            elif op == "ghz_step" and len(parts) == 2:
                q0 = int(parts[1][1:])
                if q0 < nq:
                    qc.h(q0)
        try:
            qc.draw(output="mpl", ax=ax_circ)
        except:
            ax_circ.text(0.5, 0.5, "Circuit\nunavailable", ha="center", va="center")
        ax_circ.set_title(f"Match {i}")
        ax_circ.axis("off")

    plt.tight_layout()
    plt.savefig(args.output_vis, dpi=300)
    plt.close(fig)
    print(f"Saved test visualization to {args.output_vis}")

# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test DDPM-based inverse QC model")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # -------------- TRAINING PARSER ----------------
    train_parser = subparsers.add_parser("train", help="Train mode")
    train_parser.add_argument("--train_json", type=str, required=True,
                              help="Path to training JSON (e.g., crypto_qc_train.json)")
    train_parser.add_argument("--val_json", type=str, required=True,
                              help="Path to validation JSON (e.g., crypto_qc_val.json)")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    train_parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    train_parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension (unused)")
    train_parser.add_argument("--max_len", type=int, default=32, help="Maximum token sequence length")
    train_parser.add_argument("--T", type=int, default=200, help="Number of diffusion steps")
    train_parser.add_argument("--save_interval", type=int, default=5,
                              help="Save sample visualizations every N epochs")
    train_parser.add_argument("--n_samples", type=int, default=3,
                              help="Number of validation samples to visualize each interval")
    train_parser.add_argument("--device", type=str,
                              default=("cuda" if torch.cuda.is_available() else "cpu"),
                              choices=["cpu", "cuda"], help="Device for training")
    train_parser.add_argument("--output_dir", type=str, default="results/diffusion",
                              help="Directory for checkpoints, logs, visuals")

    # -------------- TESTING PARSER ----------------
    test_parser = subparsers.add_parser("test", help="Test mode: find matching circuits")
    test_parser.add_argument("--test_json", type=str, required=True,
                             help="Path to test JSON (e.g., crypto_qc_test.json)")
    test_parser.add_argument("--model_path", type=str, required=True,
                             help="Path to a saved checkpoint (ddpm_best.pt)")
    test_parser.add_argument("--hidden_dim", type=int, default=256, help="Must match training")
    test_parser.add_argument("--latent_dim", type=int, default=128, help="Unused")
    test_parser.add_argument("--max_len", type=int, default=32, help="Must match training")
    test_parser.add_argument("--T", type=int, default=200, help="Must match training")
    test_parser.add_argument("--target_index", type=int, default=None,
                             help="Index in test set to use (if not given, pick at random)")
    test_parser.add_argument("--num_matches", type=int, default=5,
                             help="Number of distinct circuits to find that match exactly")
    test_parser.add_argument("--max_attempts", type=int, default=5000,
                             help="Max sampling attempts before giving up")
    test_parser.add_argument("--output_vis", type=str, default="diffusion_test_vis.png",
                             help="Filename for the final composite visualization")
    test_parser.add_argument("--device", type=str,
                             default=("cuda" if torch.cuda.is_available() else "cpu"),
                             choices=["cpu", "cuda"], help="Device for inference")

    args = parser.parse_args()
    if args.mode == "train":
        train_loop(args)
    elif args.mode == "test":
        test_loop(args)
    else:
        raise ValueError("Unknown mode. Use 'train' or 'test'.")
