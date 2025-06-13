#!/usr/bin/env python3
"""
train_graphvae_inverse.py

Train or test a GraphVAE-based inverse quantum circuit synthesis model on the stratified dataset.

Usage (training):
  python train_graphvae_inverse.py train \
      --train_json data/crypto_qc_train.json \
      --val_json   data/crypto_qc_val.json \
      --epochs 100 \
      --batch_size 32 \
      --lr 1e-3 \
      --hidden_dim 256 \
      --latent_dim 128 \
      --num_gcn_layers 4 \
      --max_len 32 \
      --p_hidden_dim 128 \
      --save_interval 5 \
      --n_samples 3 \
      --output_dir graphvae_results

Usage (testing):
  python train_graphvae_inverse.py test \
      --test_json data/crypto_qc_test.json \
      --model_path graphvae_results/checkpoints/graphvae_best.pt \
      --hidden_dim 256 \
      --latent_dim 128 \
      --num_gcn_layers 4 \
      --max_len 32 \
      --p_hidden_dim 128 \
      --target_index 10 \
      --num_matches 5 \
      --max_attempts 5000 \
      --output_vis graphvae_test_vis.png \
      --device cuda
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
# Tokenizer
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
        ids = [self.vocab.get(t, self.vocab["PAD"]) for t in seq][:max_len]
        if len(ids) < max_len:
            ids += [self.vocab["PAD"]] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        toks = []
        for i in ids:
            tok = self.inv_vocab.get(i, "PAD")
            if tok == "PAD":
                break
            toks.append(tok)
        return toks

    def __len__(self):
        return len(self.vocab)

# ------------------------------------------------------------
# QuantumDataset + custom collate_fn
# ------------------------------------------------------------
class QuantumDataset(Dataset):
    """
    PyTorch Dataset for inverse QC: returns (node_ids, adjacency, p_target, num_qubits, raw_probs).
    """
    def __init__(self, json_path: str, tokenizer: Tokenizer, max_len: int = 32):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_states = 2 ** 6  # always pad to 64

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        tokens = entry["tokens"]
        nq = entry["num_qubits"]
        raw_probs = entry["probs"]

        # 1) Truncate or pad token sequence
        L = len(tokens)
        if L >= self.max_len:
            tokens_proc = tokens[: self.max_len]
        else:
            tokens_proc = tokens + ["PAD"] * (self.max_len - L)

        node_ids = torch.LongTensor(self.tokenizer.encode(tokens_proc, self.max_len))

        # 2) Build adjacency matrix
        def parse_qubits(tk):
            qs = set()
            parts = tk.split()
            for p in parts:
                if p.startswith("q"):
                    try:
                        qs.add(int(p[1:]))
                    except:
                        pass
            return qs

        qubit_sets = [parse_qubits(tk) if i < L else set() for i, tk in enumerate(tokens_proc)]
        adj = np.zeros((self.max_len, self.max_len), dtype=np.float32)
        for i in range(self.max_len):
            for j in range(i + 1, self.max_len):
                if j == i + 1:
                    adj[i, j] = adj[j, i] = 1.0
                elif qubit_sets[i] & qubit_sets[j]:
                    adj[i, j] = adj[j, i] = 1.0

        # 3) Build p_target vector of length 64
        pvec = np.zeros(2 ** nq, dtype=np.float32)
        for i_state in range(2 ** nq):
            bitstr = format(i_state, f"0{nq}b")
            pvec[i_state] = raw_probs.get(bitstr, 0.0)
        if 2 ** nq < self.max_states:
            pvec = np.pad(pvec, (0, self.max_states - 2 ** nq))
        p_target = torch.FloatTensor(pvec)

        return node_ids, torch.FloatTensor(adj), p_target, nq, raw_probs

def collate_fn(batch):
    node_ids, adjs, p_targets, nqs, raw_list = zip(*batch)
    node_ids = torch.stack(node_ids, dim=0)
    adjs = torch.stack(adjs, dim=0)
    p_targets = torch.stack(p_targets, dim=0)
    nqs = torch.tensor(nqs, dtype=torch.long)
    return node_ids, adjs, p_targets, nqs, list(raw_list)

# ------------------------------------------------------------
# GraphVAE Model
# ------------------------------------------------------------
class GraphVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_gcn_layers: int = 4,
        max_len: int = 32,
        p_hidden_dim: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_gcn = num_gcn_layers
        self.max_len = max_len

        self.node_embed = nn.Embedding(vocab_size, hidden_dim)

        self.p_proj = nn.Sequential(
            nn.Linear(64, p_hidden_dim),
            nn.ReLU(),
            nn.Linear(p_hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))

        self.mu_mlp = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_mlp = nn.Linear(hidden_dim * 2, latent_dim)

        self.zp_to_h = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim * max_len),
            nn.ReLU(),
        )
        self.node_recon = nn.Linear(hidden_dim, vocab_size)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, node_ids, adj, p_target):
        B = node_ids.size(0)
        H = self.node_embed(node_ids)  # (B, max_len, hidden_dim)
        p_h = self.p_proj(p_target)    # (B, hidden_dim)
        p_h_exp = p_h.unsqueeze(1).repeat(1, self.max_len, 1)  # (B, max_len, hidden_dim)
        H = torch.cat([H, p_h_exp], dim=-1)  # (B, max_len, 2*hidden_dim)

        I = torch.eye(self.max_len, device=adj.device).unsqueeze(0).repeat(B, 1, 1)
        A_hat = adj + I
        D_hat = A_hat.sum(dim=-1)
        D_inv = torch.diag_embed(1.0 / (D_hat + 1e-8))
        A_norm = D_inv @ A_hat  # (B, max_len, max_len)

        for gcn in self.gcn_layers:
            H_gcn = A_norm @ H[..., :self.hidden_dim]  # (B, max_len, hidden_dim)
            H_cat = torch.cat([H_gcn, p_h_exp], dim=-1)  # (B, max_len, 2*hidden_dim)
            H_new = gcn(H_cat)
            H_new = F.relu(H_new)
            p_h_exp = p_h.unsqueeze(1).repeat(1, self.max_len, 1)
            H = torch.cat([H_new, p_h_exp], dim=-1)

        H_final = H[..., :self.hidden_dim]  # (B, max_len, hidden_dim)
        g_emb = H_final.sum(dim=1)         # (B, hidden_dim)

        gp = torch.cat([g_emb, p_h], dim=1)  # (B, 2*hidden_dim)
        mu = self.mu_mlp(gp)        # (B, latent_dim)
        logvar = self.logvar_mlp(gp)  # (B, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, p_target):
        B = z.size(0)
        p_h = self.p_proj(p_target)  # (B, hidden_dim)
        zp = torch.cat([z, p_h], dim=1)  # (B, latent_dim + hidden_dim)

        H_dec = self.zp_to_h(zp)  # (B, hidden_dim * max_len)
        H_dec = H_dec.view(B, self.max_len, self.hidden_dim)  # (B, max_len, hidden_dim)

        node_logits = self.node_recon(H_dec)  # (B, max_len, vocab_size)

        H_i = H_dec.unsqueeze(2).repeat(1, 1, self.max_len, 1)
        H_j = H_dec.unsqueeze(1).repeat(1, self.max_len, 1, 1)
        H_pair = torch.cat([H_i, H_j], dim=-1)  # (B, max_len, max_len, 2*hidden_dim)
        edge_logits = self.edge_mlp(H_pair).squeeze(-1)  # (B, max_len, max_len)

        return node_logits, edge_logits

    def forward(self, node_ids, adj, p_target):
        mu, logvar = self.encode(node_ids, adj, p_target)
        z = self.reparameterize(mu, logvar)
        node_logits, edge_logits = self.decode(z, p_target)
        return node_logits, edge_logits, mu, logvar

# ------------------------------------------------------------
# Reconstruction and simulation helpers
# ------------------------------------------------------------
def reconstruct_graph(node_logits, edge_logits, threshold=0.5):
    node_ids_pred = node_logits.argmax(dim=-1).cpu().tolist()  # (max_len)
    edge_probs = torch.sigmoid(edge_logits).cpu().detach().numpy()      # (max_len, max_len)
    adj_pred = (edge_probs > threshold).astype(np.int32)
    return node_ids_pred, adj_pred

def decode_tokens(node_ids_pred, tokenizer: Tokenizer):
    return tokenizer.decode(node_ids_pred)

def simulate_qc(tokens, nq):
    qc = QuantumCircuit(nq)
    for tok in tokens:
        parts = tok.split()
        if parts[0] in {"h", "x", "y", "z"} and len(parts) == 2:
            gate = parts[0]; q = int(parts[1][1:])
            if q < nq:
                getattr(qc, gate)(q)
        elif parts[0] == "cx" and len(parts) == 3:
            q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
            if q1 < nq and q2 < nq:
                qc.cx(q1, q2)
        elif parts[0] in {"rx", "ry", "rz"} and len(parts) == 2:
            gate_name = parts[0][:2]
            bucket = int(parts[0][3:-1])
            angle = bucket * 2 * np.pi / 8
            q = int(parts[1][1:])
            if q < nq:
                getattr(qc, gate_name)(angle, q)
        elif parts[0] in {"e91_entangle", "ghz_entangle"} and len(parts) == 3:
            q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
            if q1 < nq and q2 < nq:
                qc.h(q1); qc.cx(q1, q2)
        elif parts[0] == "ghz_step" and len(parts) == 2:
            q0 = int(parts[1][1:])
            if q0 < nq:
                qc.h(q0)
    sv = Statevector.from_instruction(qc)
    return sv.probabilities_dict()

def distribution_matching_loss(tokens_pred, nq, p_target_vec):
    try:
        pr_dict = simulate_qc(tokens_pred, nq)
    except:
        return torch.tensor(50.0, device=p_target_vec.device)
    states = [format(i, f"0{nq}b") for i in range(2 ** nq)]
    pr_vec = torch.tensor([pr_dict.get(s, 0.0) for s in states], device=p_target_vec.device)
    if pr_vec.sum() == 0:
        return torch.tensor(50.0, device=p_target_vec.device)
    pr_vec = pr_vec / pr_vec.sum()
    gt_vec = p_target_vec[: 2 ** nq]
    gt_vec = gt_vec / (gt_vec.sum() + 1e-10)
    mask = gt_vec > 0
    kl = torch.sum(gt_vec[mask] * torch.log(gt_vec[mask] / (pr_vec[mask] + 1e-10)))
    return kl

# ------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------
def reconstruction_loss(node_logits, edge_logits, node_ids, adj):
    B, N, V = node_logits.size()
    node_logits_flat = node_logits.view(B * N, V)
    node_ids_flat = node_ids.view(B * N)
    ce_node = F.cross_entropy(node_logits_flat, node_ids_flat, ignore_index=0, reduction="sum")

    edge_logits_flat = edge_logits.view(B * N * N)
    adj_flat = adj.view(B * N * N)
    bce_edge = F.binary_cross_entropy_with_logits(edge_logits_flat, adj_flat, reduction="sum")

    return ce_node + bce_edge

def kl_divergence_z(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# ------------------------------------------------------------
# Utility: Save sample visualizations during training
# ------------------------------------------------------------
def save_sample_vis(model, dataset, tokenizer, epoch, out_dir, device, n_samples=3):
    """
    Pick n_samples random validation entries; for each:
      - Encode → sample z → decode → reconstruct tokens
      - Simulate and compare GT vs generated distributions
      - Plot circuit diagram + full-spectrum bar chart
    """
    chosen = random.sample(range(len(dataset)), n_samples)
    for i, idx in enumerate(chosen, start=1):
        node_ids_gt, adj_gt, p_target, nq, raw_probs = dataset[idx]
        p_target = p_target.to(device)
        nq_tensor = torch.tensor(nq, device=device)

        # Encode to get mu, logvar, then sample z
        model.eval()
        with torch.no_grad():
            mu, logvar = model.encode(node_ids_gt.unsqueeze(0).to(device),
                                      adj_gt.unsqueeze(0).to(device),
                                      p_target.unsqueeze(0))
            z = model.reparameterize(mu, logvar)
            node_logits, edge_logits = model.decode(z, p_target.unsqueeze(0))

        node_logits_0 = node_logits[0]
        edge_logits_0 = edge_logits[0]
        node_ids_pred, _ = reconstruct_graph(node_logits_0, edge_logits_0)
        gen_tokens = decode_tokens(node_ids_pred, tokenizer)

        try:
            pred_dict = simulate_qc(gen_tokens, nq)
        except:
            pred_dict = {}

        states = [format(x, f"0{nq}b") for x in range(2 ** nq)]
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
        ax_bar.bar(x + 0.15, pr_vals, width=0.3, label="Generated", color="orange")
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
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_fn)

    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu")

    model = GraphVAE(
        vocab_size=len(tokenizer),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_gcn_layers=args.num_gcn_layers,
        max_len=args.max_len,
        p_hidden_dim=args.p_hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logs = {
        "epoch": [],
        "train_loss": [],
        "val_loss": []
    }
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for node_ids, adj, p_target, nq, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            node_ids = node_ids.to(device)
            adj = adj.to(device)
            p_target = p_target.to(device)
            nq = nq.to(device)

            optimizer.zero_grad()
            node_logits, edge_logits, mu, logvar = model(node_ids, adj, p_target)
            recon_loss = reconstruction_loss(node_logits, edge_logits, node_ids, adj)
            kl_loss = kl_divergence_z(mu, logvar)

            # Auxiliary distribution-matching on first sample in batch
            node_logits_0 = node_logits[0]
            edge_logits_0 = edge_logits[0]
            node_ids_pred, _ = reconstruct_graph(node_logits_0, edge_logits_0)
            tokens_pred = decode_tokens(node_ids_pred, tokenizer)
            dist_loss = distribution_matching_loss(tokens_pred, nq[0].item(), p_target[0])

            total_loss = recon_loss + kl_loss + 10.0 * dist_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        avg_train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for node_ids, adj, p_target, nq, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                node_ids = node_ids.to(device)
                adj = adj.to(device)
                p_target = p_target.to(device)
                nq = nq.to(device)

                node_logits, edge_logits, mu, logvar = model(node_ids, adj, p_target)
                recon_loss = reconstruction_loss(node_logits, edge_logits, node_ids, adj)
                kl_loss = kl_divergence_z(mu, logvar)

                node_logits_0 = node_logits[0]
                edge_logits_0 = edge_logits[0]
                node_ids_pred, _ = reconstruct_graph(node_logits_0, edge_logits_0)
                tokens_pred = decode_tokens(node_ids_pred, tokenizer)
                dist_loss = distribution_matching_loss(tokens_pred, nq.item(), p_target[0])

                total_val = recon_loss + kl_loss + 10.0 * dist_loss
                val_loss_accum += total_val.item()

        avg_val_loss = val_loss_accum / len(val_ds)
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        logs["epoch"].append(epoch)
        logs["train_loss"].append(avg_train_loss)
        logs["val_loss"].append(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "num_gcn_layers": args.num_gcn_layers,
                "max_len": args.max_len,
                "p_hidden_dim": args.p_hidden_dim,
                "vocab_size": len(tokenizer),
                "epoch": epoch,
            }
            ckpt_path = ckpt_dir / "graphvae_best.pt"
            torch.save(ckpt, ckpt_path)
            print(f"  → New best model saved: Val Loss {best_val_loss:.4f}")

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
      - Load trained checkpoint
      - Load test JSON
      - Select a target distribution by index or random
      - Sample latent z ~ N(0,I), decode to tokens, simulate
      - Repeat until num_matches circuits match exactly
      - Plot one figure: left=target distribution, right=subplots of matching circuits
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    tokenizer = Tokenizer(max_qubits=6)
    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)

    model = GraphVAE(
        vocab_size=checkpoint["vocab_size"],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_gcn_layers=args.num_gcn_layers,
        max_len=args.max_len,
        p_hidden_dim=args.p_hidden_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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
        z = torch.randn(1, args.latent_dim, device=device)
        with torch.no_grad():
            node_logits, edge_logits = model.decode(z, p_target_tensor.unsqueeze(0))

        node_logits_0 = node_logits[0]
        edge_logits_0 = edge_logits[0]
        node_ids_pred, _ = reconstruct_graph(node_logits_0, edge_logits_0)
        gen_tokens = decode_tokens(node_ids_pred, tokenizer)

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
    parser = argparse.ArgumentParser(description="Train or test GraphVAE-based inverse QC model")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # -------------- TRAINING PARSER ----------------
    train_parser = subparsers.add_parser("train", help="Train mode")
    train_parser.add_argument("--train_json", type=str, required=True,
                              help="Path to training JSON (e.g., crypto_qc_train.json)")
    train_parser.add_argument("--val_json", type=str, required=True,
                              help="Path to validation JSON (e.g., crypto_qc_val.json)")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    train_parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension")
    train_parser.add_argument("--num_gcn_layers", type=int, default=4, help="Number of GCN layers")
    train_parser.add_argument("--max_len", type=int, default=32, help="Maximum token sequence length")
    train_parser.add_argument("--p_hidden_dim", type=int, default=128, help="Hidden dim for p_target projection")
    train_parser.add_argument("--save_interval", type=int, default=5,
                              help="Save sample visualizations every N epochs")
    train_parser.add_argument("--n_samples", type=int, default=3,
                              help="Number of validation samples to visualize at each interval")
    train_parser.add_argument("--device", type=str,
                              default=("cuda" if torch.cuda.is_available() else "cpu"),
                              choices=["cpu", "cuda"], help="Device for training")
    train_parser.add_argument("--output_dir", type=str, default="results/graphvae",
                              help="Directory for checkpoints, logs, visuals")

    # -------------- TESTING PARSER ----------------
    test_parser = subparsers.add_parser("test", help="Test mode: find matching circuits")
    test_parser.add_argument("--test_json", type=str, required=True,
                             help="Path to test JSON (e.g., crypto_qc_test.json)")
    test_parser.add_argument("--model_path", type=str, required=True,
                             help="Path to a saved .pt checkpoint (graphvae_best.pt)")
    test_parser.add_argument("--hidden_dim", type=int, default=256, help="Must match training")
    test_parser.add_argument("--latent_dim", type=int, default=128, help="Must match training")
    test_parser.add_argument("--num_gcn_layers", type=int, default=4, help="Must match training")
    test_parser.add_argument("--max_len", type=int, default=32, help="Must match training")
    test_parser.add_argument("--p_hidden_dim", type=int, default=128, help="Must match training")
    test_parser.add_argument("--target_index", type=int, default=None,
                             help="Index in test set to use (if not given, pick at random)")
    test_parser.add_argument("--num_matches", type=int, default=5,
                             help="Number of distinct circuits to find that match exactly")
    test_parser.add_argument("--max_attempts", type=int, default=5000,
                             help="Max sampling attempts before giving up")
    test_parser.add_argument("--output_vis", type=str, default="graphvae_test_vis.png",
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
