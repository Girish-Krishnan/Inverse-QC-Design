#!/usr/bin/env python3
"""
train_transformer_inverse.py

Train or test a Transformer-based inverse quantum circuit synthesis model on the stratified dataset.

Usage (training):
  python train_transformer_inverse.py \
      train \
      --train_json data/crypto_qc_train.json \
      --val_json   data/crypto_qc_val.json \
      --epochs 100 \
      --batch_size 32 \
      --lr 1e-3 \
      --save_interval 5 \
      --output_dir transformer_results

Usage (testing):
  python train_transformer_inverse.py \
      test \
      --test_json data/crypto_qc_test.json \
      --model_path transformer_results/checkpoints/best_model.pt \
      --target_index 10 \
      --num_matches 5 \
      --output_vis transformer_test_vis.png \
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
from torch.utils.data import Dataset, DataLoader

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
        gates = ['h', 'x', 'y', 'z', 'cx', 'rx', 'ry', 'rz']
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

    def encode(self, token_list, max_len: int = 32):
        """
        Encode a sequence of token strings into a list of IDs, padded/truncated to max_len.
        Inserts START at position 0, END at final used position, PAD thereafter.
        """
        ids = [self.vocab.get(t, self.vocab["PAD"]) for t in token_list]
        ids = ids[: max_len - 2]
        padded = [self.vocab["START"], *ids, self.vocab["END"]]
        if len(padded) < max_len:
            padded += [self.vocab["PAD"]] * (max_len - len(padded))
        return padded

    def decode(self, id_list):
        """
        Decode a list of IDs into token strings, skipping PAD/START/END.
        """
        special = {self.vocab["PAD"], self.vocab["START"], self.vocab["END"]}
        return [self.inv_vocab[i] for i in id_list if i not in special]

    def __len__(self):
        return len(self.vocab)

# ------------------------------------------------------------
# QuantumDataset + custom collate_fn
# ------------------------------------------------------------
class QuantumDataset(Dataset):
    """
    PyTorch Dataset for inverse QC: returns (p_target, token_ids, num_qubits, raw_probs_dict).
    """
    def __init__(self, json_path: str, tokenizer: Tokenizer, max_len: int = 32):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
          p_target: FloatTensor of length 64 (padded to 6-qubit support)
          token_ids: LongTensor of length max_len
          num_qubits: LongTensor scalar
          raw_probs: dict mapping bitstrings to probabilities (used for loss/visuals)
        """
        entry = self.data[idx]
        nq = entry["num_qubits"]
        raw_probs = entry["probs"]
        # Build p_target as float32 array length 2^nq, then pad to 64
        p = np.zeros(2 ** nq, dtype=np.float32)
        for i in range(2 ** nq):
            bit = format(i, f"0{nq}b")
            p[i] = raw_probs.get(bit, 0.0)
        if p.shape[0] < 64:
            p = np.pad(p, (0, 64 - p.shape[0]), constant_values=0.0)
        p_target = torch.tensor(p, dtype=torch.float32)

        token_ids = torch.tensor(self.tokenizer.encode(entry["tokens"], self.max_len),
                                 dtype=torch.long)
        return p_target, token_ids, torch.tensor(nq, dtype=torch.long), raw_probs

def collate_fn(batch):
    """
    Custom collate function:
      - Batches p_target, token_ids, num_qubits into tensors
      - Leaves raw_probs as a list of dictionaries
    """
    p_targets, token_ids, nqs, raw_list = zip(*batch)
    p_targets = torch.stack(p_targets, dim=0)
    token_ids = torch.stack(token_ids, dim=0)
    nqs       = torch.stack(nqs, dim=0)
    return p_targets, token_ids, nqs, list(raw_list)

# ------------------------------------------------------------
# Transformer Encoder-Decoder Model
# ------------------------------------------------------------
class InverseGenTransformer(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int, hidden: int = 128,
                 nhead: int = 4, num_layers: int = 4, max_len: int = 32):
        super().__init__()
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.token_embed = nn.Embedding(vocab_size, hidden)
        self.pos_embed = nn.Embedding(max_len, hidden)
        self.transformer = nn.Transformer(
            d_model=hidden,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden, vocab_size)
        self.max_len = max_len

    def forward(self, probs, num_qubits, tgt_ids):
        """
        probs: (B, 64)
        num_qubits: (B,)
        tgt_ids: (B, T)
        Returns logits: (B, T, vocab_size)
        """
        B, T = tgt_ids.size()
        nq_norm = num_qubits.float().div(6.0).unsqueeze(1)       # (B,1)
        enc_input = self.encoder_fc(torch.cat([probs, nq_norm], dim=1))  # (B, hidden)
        enc_input = enc_input.unsqueeze(1).repeat(1, T, 1)       # (B, T, hidden)

        pos_idx = torch.arange(T, device=tgt_ids.device).unsqueeze(0)  # (1, T)
        dec_in = self.token_embed(tgt_ids) + self.pos_embed(pos_idx)   # (B, T, hidden)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_ids.device)
        out = self.transformer(enc_input, dec_in, tgt_mask=tgt_mask)
        return self.fc_out(out)

    def generate_greedy(self, probs, num_qubits, sos_id, eos_id, max_len=None, device="cpu"):
        """
        Greedy decoding: returns token IDs including SOS, excluding final EOS.
        """
        if max_len is None:
            max_len = self.max_len
        seq = [sos_id]
        for _ in range(max_len - 1):
            inp = torch.tensor([seq], dtype=torch.long, device=device)  # (1, t)
            logits = self.forward(probs.unsqueeze(0), num_qubits.unsqueeze(0), inp)  # (1, t, V)
            next_tok = logits[0, -1].argmax().item()
            if next_tok == eos_id:
                break
            seq.append(next_tok)
        return seq

    def generate_sample(self, probs, num_qubits, sos_id, eos_id, max_len=None, device="cpu"):
        """
        Stochastic sampling decoding: returns token IDs including SOS, excluding final EOS.
        """
        if max_len is None:
            max_len = self.max_len
        seq = [sos_id]
        for _ in range(max_len - 1):
            inp = torch.tensor([seq], dtype=torch.long, device=device)  # (1, t)
            logits = self.forward(probs.unsqueeze(0), num_qubits.unsqueeze(0), inp)  # (1, t, V)
            prob_dist = torch.softmax(logits[0, -1], dim=-1).cpu().numpy()  # (V,)
            nxt = np.random.choice(len(prob_dist), p=prob_dist)
            if nxt == eos_id:
                break
            seq.append(nxt)
        return seq

# ------------------------------------------------------------
# Physics-based losses + simulation
# ------------------------------------------------------------
def kl_divergence(p, q):
    """KL(p || q) for distributions p,q as tensors. Assumes p>0 where applicable."""
    mask = p > 0
    return float((p[mask] * torch.log(p[mask] / (q[mask] + 1e-12))).sum().item())

def entanglement_entropy_loss(true_probs, pred_probs, num_qubits):
    """
    Compute |S_true - S_pred| where S = -Tr(rho_sub log rho_sub).
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

def simulate(tokens, num_qubits):
    """
    Build a Qiskit QuantumCircuit from token strings; compute probabilities_dict.
    Bucketed rotations: angle = (bucket/8)*2π.
    """
    n = num_qubits
    qc = QuantumCircuit(n)
    for tok in tokens:
        parts = tok.split()
        op = parts[0]
        if op in {'h','x','y','z'} and len(parts) == 2:
            q = int(parts[1][1:])
            if q < n:
                getattr(qc, op)(q)
        elif op == 'cx' and len(parts) == 3:
            q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
            if q1 < n and q2 < n:
                qc.cx(q1, q2)
        elif op.startswith(('rx','ry','rz')) and len(parts) == 2:
            gate = op[:2]
            try:
                bucket = int(op[3:-1])
                angle = bucket * 2 * np.pi / 8
                q = int(parts[1][1:])
                if q < n:
                    getattr(qc, gate)(angle, q)
            except:
                continue
        elif op == 'e91_entangle' and len(parts) == 3:
            q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
            if q1 < n and q2 < n:
                qc.h(q1); qc.cx(q1, q2)
        elif op == 'ghz_entangle' and len(parts) == 3:
            q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
            if q1 == 0 and q2 < n:
                qc.cx(0, q2)
        elif op == 'ghz_step' and len(parts) == 2:
            q0 = int(parts[1][1:])
            if q0 == 0 and n > 0:
                qc.h(0)
        # skip bb84_step, unknown tokens
    sv = Statevector.from_instruction(qc)
    return sv.probabilities_dict()

# ------------------------------------------------------------
# Utility: Save a sample visualization during training
# ------------------------------------------------------------
def save_sample_vis(model, dataset, tokenizer, epoch, out_dir, device, n_samples=3):
    """
    Pick n_samples random indices from `dataset` (usually validation set),
    run inference (greedy decode), simulate, and save a figure per sample:
      - Left: generated circuit diagram
      - Right: bar chart comparing ground truth vs generated distributions
    Filenames: epoch<XXX>_sample<i>.png
    """
    sos_id = tokenizer.vocab["START"]
    eos_id = tokenizer.vocab["END"]

    chosen = random.sample(range(len(dataset)), n_samples)
    for i, idx in enumerate(chosen, start=1):
        p_target, token_ids, nq_tensor, gt_probs = dataset[idx]
        p_target = p_target.to(device)
        nq_tensor = nq_tensor.to(device)

        # Generate sequence (greedy)
        gen_ids = model.generate_greedy(p_target, nq_tensor,
                                        sos_id, eos_id,
                                        max_len=tokenizer.__len__(), device=device)
        gen_tokens = tokenizer.decode(gen_ids)

        # Simulate generated
        try:
            pred_dict = simulate(gen_tokens, int(nq_tensor.item()))
        except:
            pred_dict = {}

        # Ground-truth dict
        gt_dict = gt_probs

        # Build full state lists
        n = int(nq_tensor.item())
        states = [format(x, f"0{n}b") for x in range(2**n)]
        gt_vals = [gt_dict.get(s, 0.0) for s in states]
        pr_vals = [pred_dict.get(s, 0.0) for s in states]

        # Plot
        fig, (ax_circ, ax_bar) = plt.subplots(1, 2, figsize=(10, 4),
                                              gridspec_kw={'width_ratios': [1, 2]})
        # Circuit diagram
        qc = QuantumCircuit(n)
        for tok in gen_tokens:
            parts = tok.split()
            op = parts[0]
            if op in {'h','x','y','z'} and len(parts) == 2:
                q = int(parts[1][1:])
                if q < n:
                    getattr(qc, op)(q)
            elif op == 'cx' and len(parts) == 3:
                q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
                if q1 < n and q2 < n:
                    qc.cx(q1, q2)
            elif op.startswith(('rx','ry','rz')) and len(parts) == 2:
                gate = op[:2]
                try:
                    bucket = int(op[3:-1])
                    angle = bucket * 2 * np.pi / 8
                    q = int(parts[1][1:])
                    if q < n:
                        getattr(qc, gate)(angle, q)
                except:
                    pass
            elif op == 'e91_entangle' and len(parts) == 3:
                q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
                if q1 < n and q2 < n:
                    qc.h(q1); qc.cx(q1, q2)
            elif op == 'ghz_entangle' and len(parts) == 3:
                q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
                if q1 == 0 and q2 < n:
                    qc.cx(0, q2)
            elif op == 'ghz_step' and len(parts) == 2:
                q0 = int(parts[1][1:])
                if q0 == 0 and n > 0:
                    qc.h(0)
        try:
            qc.draw(output='mpl', ax=ax_circ)
        except:
            ax_circ.text(0.5, 0.5, "Circuit\nunavailable", ha='center', va='center')
        ax_circ.set_title(f"Epoch {epoch}  Sample {i}  ({n} qubits)")
        ax_circ.axis('off')

        # Bar chart full spectrum
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

    # Create output directories
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    vis_dir  = out_dir / "visuals"
    ckpt_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    # Initialize tokenizer and datasets
    tokenizer = Tokenizer(max_qubits=6)
    train_ds = QuantumDataset(args.train_json, tokenizer, max_len=args.max_len)
    val_ds   = QuantumDataset(args.val_json, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,
                              batch_size=1,
                              shuffle=False,
                              num_workers=0,
                              collate_fn=collate_fn)

    # Instantiate model
    model = InverseGenTransformer(input_dim=64,
                                  vocab_size=len(tokenizer),
                                  hidden=args.hidden_dim,
                                  nhead=args.nhead,
                                  num_layers=args.num_layers,
                                  max_len=args.max_len)
    device = torch.device("cuda" if (torch.cuda.is_available() and args.device=="cuda") else "cpu")
    model.to(device)

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["PAD"])

    # Containers for logging
    logs = {
        "epoch": [],
        "train_xent": [],
        "train_kl": [],
        "train_ent": [],
        "val_xent": [],
        "val_kl": [],
        "val_ent": []
    }
    best_val_kl = float("inf")

    # Training epochs
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_xent = 0.0
        running_kl   = 0.0
        running_ent  = 0.0
        total_batches = 0

        for p_target, token_ids, nq_tensor, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            p_target = p_target.to(device)
            token_ids = token_ids.to(device)
            nq_tensor = nq_tensor.to(device)

            optimizer.zero_grad()
            # Input to decoder: all tokens except last
            inp_ids = token_ids[:, :-1]       # (B, T-1)
            logits = model(p_target, nq_tensor, inp_ids)   # (B, T-1, V)

            # Cross-entropy loss
            tgt_ids = token_ids[:, 1:].reshape(-1)        # (B*(T-1),)
            xent_loss = ce_loss_fn(logits.reshape(-1, logits.size(-1)), tgt_ids)

            # Physics-based losses: compute on a small random subset of the batch
            batch_size = p_target.size(0)
            subset_idx = random.sample(range(batch_size), min(batch_size, args.physics_samples))
            kl_losses = []
            ent_losses = []
            with torch.no_grad():
                for i in subset_idx:
                    # Greedy decode for sample i
                    seq_ids = model.generate_greedy(p_target[i], nq_tensor[i],
                                                     tokenizer.vocab["START"],
                                                     tokenizer.vocab["END"],
                                                     max_len=args.max_len,
                                                     device=device)
                    gen_tokens = tokenizer.decode(seq_ids)
                    # Simulate generated
                    try:
                        pred_dict = simulate(gen_tokens, int(nq_tensor[i].item()))
                        # Build pred_probs tensor
                        n = int(nq_tensor[i].item())
                        pred_p = torch.zeros(64, device=device)
                        for j in range(2**n):
                            bit = format(j, f"0{n}b")
                            pred_p[j] = pred_dict.get(bit, 0.0)
                        true_p = p_target[i]
                        kl_val = kl_divergence(true_p, pred_p)
                        ent_val = entanglement_entropy_loss(true_p, pred_p, nq_tensor[i])
                        kl_losses.append(kl_val)
                        ent_losses.append(ent_val.item())
                    except:
                        continue

            if kl_losses:
                avg_kl = np.mean(kl_losses)
                avg_ent = np.mean(ent_losses)
            else:
                avg_kl = 0.0
                avg_ent = 0.0

            total_loss = xent_loss + args.lambda_kl * avg_kl + args.lambda_ent * avg_ent
            total_loss.backward()
            optimizer.step()

            running_xent += xent_loss.item()
            running_kl   += avg_kl
            running_ent  += avg_ent
            total_batches += 1

        train_xent_epoch = running_xent / total_batches
        train_kl_epoch   = running_kl   / total_batches
        train_ent_epoch  = running_ent  / total_batches

        # Validation
        model.eval()
        val_running_xent = 0.0
        val_running_kl   = 0.0
        val_running_ent  = 0.0
        val_batches = 0

        with torch.no_grad():
            for p_target, token_ids, nq_tensor, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                p_target = p_target.to(device)
                token_ids = token_ids.to(device)
                nq_tensor = nq_tensor.to(device)

                inp_ids = token_ids[:, :-1]
                logits = model(p_target, nq_tensor, inp_ids)
                tgt_ids = token_ids[:, 1:].reshape(-1)
                xent_loss = ce_loss_fn(logits.reshape(-1, logits.size(-1)), tgt_ids)

                # Physics for single sample
                seq_ids = model.generate_greedy(p_target[0], nq_tensor[0],
                                                 tokenizer.vocab["START"],
                                                 tokenizer.vocab["END"],
                                                 max_len=args.max_len,
                                                 device=device)
                gen_tokens = tokenizer.decode(seq_ids)
                try:
                    pred_dict = simulate(gen_tokens, int(nq_tensor[0].item()))
                    n = int(nq_tensor[0].item())
                    pred_p = torch.zeros(64, device=device)
                    for j in range(2**n):
                        bit = format(j, f"0{n}b")
                        pred_p[j] = pred_dict.get(bit, 0.0)
                    true_p = p_target[0]
                    kl_val = kl_divergence(true_p, pred_p)
                    ent_val = entanglement_entropy_loss(true_p, pred_p, nq_tensor[0])
                except:
                    kl_val = 0.0
                    ent_val = 0.0

                val_running_xent += xent_loss.item()
                val_running_kl   += kl_val
                val_running_ent  += ent_val.item()
                val_batches += 1

        val_xent_epoch = val_running_xent / val_batches
        val_kl_epoch   = val_running_kl   / val_batches
        val_ent_epoch  = val_running_ent  / val_batches

        # Log epoch
        logs["epoch"].append(epoch)
        logs["train_xent"].append(train_xent_epoch)
        logs["train_kl"].append(train_kl_epoch)
        logs["train_ent"].append(train_ent_epoch)
        logs["val_xent"].append(val_xent_epoch)
        logs["val_kl"].append(val_kl_epoch)
        logs["val_ent"].append(val_ent_epoch)

        print(f"Epoch {epoch:03d} | "
              f"Train: XENT={train_xent_epoch:.4f}, KL={train_kl_epoch:.4f}, ENT={train_ent_epoch:.4f} | "
              f"Val:   XENT={val_xent_epoch:.4f}, KL={val_kl_epoch:.4f}, ENT={val_ent_epoch:.4f}")

        # Save best model by validation KL
        if val_kl_epoch < best_val_kl:
            best_val_kl = val_kl_epoch
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → New best model saved to {ckpt_path}")

        # Save periodic samples
        if epoch % args.save_interval == 0:
            epoch_vis_dir = vis_dir / f"epoch_{epoch:03d}"
            epoch_vis_dir.mkdir(exist_ok=True)
            save_sample_vis(model, val_ds, tokenizer, epoch, epoch_vis_dir, device, n_samples=args.n_samples)

        # Save logs to JSON
        with open(out_dir / "training_logs.json", "w") as lf:
            json.dump(logs, lf, indent=2)

    print("Training complete.")

# ------------------------------------------------------------
# Testing routine
# ------------------------------------------------------------
def test_loop(args):
    """
    In test mode, we:
      - Load a trained checkpoint
      - Load the test JSON
      - Pick one target distribution (by index or random)
      - Repeatedly sample circuits (stochastic decoding) until we find `num_matches`
        distinct circuits whose simulated distribution exactly matches the target.
      - Finally, plot a single figure:
            • left: target distribution (full 2^n states)
            • right: the `num_matches` circuit diagrams side by side.
      - Save that figure to `--output_vis`.
    """
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load tokenizer + model
    tokenizer = Tokenizer(max_qubits=6)
    device = torch.device("cuda" if (torch.cuda.is_available() and args.device=="cuda") else "cpu")
    model = InverseGenTransformer(input_dim=64,
                                  vocab_size=len(tokenizer),
                                  hidden=args.hidden_dim,
                                  nhead=args.nhead,
                                  num_layers=args.num_layers,
                                  max_len=args.max_len)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    # Load test JSON
    with open(args.test_json, "r") as f:
        test_data = json.load(f)
    test_ds = []
    for entry in test_data:
        nq = entry["num_qubits"]
        raw_probs = entry["probs"]
        # Build p_target (padded to 64)
        p = np.zeros(2 ** nq, dtype=np.float32)
        for i in range(2 ** nq):
            bit = format(i, f"0{nq}b")
            p[i] = raw_probs.get(bit, 0.0)
        if p.shape[0] < 64:
            p = np.pad(p, (0, 64 - p.shape[0]), constant_values=0.0)
        test_ds.append((torch.tensor(p, dtype=torch.float32), nq, raw_probs))

    # Select target index
    if args.target_index is not None and 0 <= args.target_index < len(test_ds):
        idx = args.target_index
    else:
        idx = random.randrange(len(test_ds))

    p_target_tensor, nq, gt_dict = test_ds[idx]
    p_target_tensor = p_target_tensor.to(device)

    # We want exactly `num_matches` distinct circuits whose simulated distribution matches gt_dict exactly
    matches = []
    attempts = 0
    seen_circuit_keys = set()

    sos_id = tokenizer.vocab["START"]
    eos_id = tokenizer.vocab["END"]

    print(f"Selected test index {idx} (n_qubits={nq}). Searching for {args.num_matches} exact matches...")

    # Precompute the ground truth dictionary (including zero probs for completeness)
    n = nq
    full_states = [format(i, f"0{n}b") for i in range(2**n)]

    while len(matches) < args.num_matches and attempts < args.max_attempts:
        attempts += 1
        # Sample one circuit via stochastic decoding
        seq_ids = model.generate_sample(p_target_tensor, torch.tensor(nq, device=device),
                                        sos_id, eos_id,
                                        max_len=args.max_len, device=device)
        gen_tokens = tokenizer.decode(seq_ids)
        # Create a uniqueness key for the token sequence + n_qubits
        key = "|".join(gen_tokens) + f"_n={nq}"
        if key in seen_circuit_keys:
            continue
        seen_circuit_keys.add(key)

        # Simulate this generated circuit
        try:
            pred_dict = simulate(gen_tokens, nq)
        except:
            continue

        # Check for exact match (needs to compare all 2^n probabilities)
        is_match = True
        for state in full_states:
            if abs(gt_dict.get(state, 0.0) - pred_dict.get(state, 0.0)) > 1e-9:
                is_match = False
                break
        if is_match:
            matches.append(gen_tokens)

    print(f"  → Found {len(matches)} matches in {attempts} attempts.")

    # Build and save visualization: bar chart + the matching circuits
    fig = plt.figure(figsize=(4 + 3*args.num_matches, 4))
    # Bar chart subplot
    ax_bar = fig.add_subplot(1, args.num_matches + 1, 1)
    gt_vals = [gt_dict.get(s, 0.0) for s in full_states]
    x = np.arange(len(full_states))
    ax_bar.bar(x, gt_vals, color="steelblue")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(full_states, rotation=90, fontsize=6)
    ax_bar.set_ylabel("Probability")
    ax_bar.set_title(f"Target Distribution (n={nq})")
    ax_bar.set_xlim(-0.5, len(full_states) - 0.5)

    # Circuit diagrams
    for i, gen_tokens in enumerate(matches, start=1):
        ax_circ = fig.add_subplot(1, args.num_matches + 1, i+1)
        qc = QuantumCircuit(nq)
        for tok in gen_tokens:
            parts = tok.split()
            op = parts[0]
            if op in {'h','x','y','z'} and len(parts) == 2:
                q = int(parts[1][1:])
                if q < nq:
                    getattr(qc, op)(q)
            elif op == 'cx' and len(parts) == 3:
                q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
                if q1 < nq and q2 < nq:
                    qc.cx(q1, q2)
            elif op.startswith(('rx','ry','rz')) and len(parts) == 2:
                gate = op[:2]
                try:
                    bucket = int(op[3:-1])
                    angle = bucket * 2 * np.pi / 8
                    q = int(parts[1][1:])
                    if q < nq:
                        getattr(qc, gate)(angle, q)
                except:
                    pass
            elif op == 'e91_entangle' and len(parts) == 3:
                q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
                if q1 < nq and q2 < nq:
                    qc.h(q1); qc.cx(q1, q2)
            elif op == 'ghz_entangle' and len(parts) == 3:
                q1 = int(parts[1][1:]); q2 = int(parts[2][1:])
                if q1 == 0 and q2 < nq:
                    qc.cx(0, q2)
            elif op == 'ghz_step' and len(parts) == 2:
                q0 = int(parts[1][1:])
                if q0 == 0 and nq > 0:
                    qc.h(0)
        try:
            qc.draw(output='mpl', ax=ax_circ)
        except:
            ax_circ.text(0.5, 0.5, "Circuit\nunavailable", ha='center', va='center')
        ax_circ.set_title(f"Match {i}")
        ax_circ.axis('off')

    plt.tight_layout()
    plt.savefig(args.output_vis, dpi=300)
    plt.close(fig)
    print(f"Saved test visualization to {args.output_vis}")

# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test Transformer-based inverse QC model")

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
    train_parser.add_argument("--hidden_dim", type=int, default=128, help="Transformer hidden dimension")
    train_parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    train_parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers")
    train_parser.add_argument("--max_len", type=int, default=32, help="Maximum token sequence length")
    train_parser.add_argument("--lambda_kl", type=float, default=1e-1,
                        help="Weight for KL divergence loss")
    train_parser.add_argument("--lambda_ent", type=float, default=5e-2,
                        help="Weight for entanglement entropy loss")
    train_parser.add_argument("--physics_samples", type=int, default=2,
                        help="Number of samples per batch for physics-based loss")
    train_parser.add_argument("--save_interval", type=int, default=5,
                        help="Save sample visualizations every N epochs")
    train_parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of validation samples to visualize at each interval")
    train_parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        choices=["cpu","cuda"], help="Device for training/inference")
    train_parser.add_argument("--output_dir", type=str, default="results/transformer",
                        help="Directory for checkpoints, logs, visuals")

    # -------------- TESTING PARSER ----------------
    test_parser = subparsers.add_parser("test", help="Test mode: find matching circuits")
    test_parser.add_argument("--test_json", type=str, required=True,
                       help="Path to test JSON (e.g., crypto_qc_test.json)")
    test_parser.add_argument("--model_path", type=str, required=True,
                       help="Path to a saved .pt checkpoint (best_model.pt)")
    test_parser.add_argument("--hidden_dim", type=int, default=128, help="Must match training")
    test_parser.add_argument("--nhead", type=int, default=4, help="Must match training")
    test_parser.add_argument("--num_layers", type=int, default=4, help="Must match training")
    test_parser.add_argument("--max_len", type=int, default=32, help="Must match training")
    test_parser.add_argument("--target_index", type=int, default=None,
                       help="Index in test set to use (if not given, pick at random)")
    test_parser.add_argument("--num_matches", type=int, default=5,
                       help="Number of distinct circuits to find that match exactly")
    test_parser.add_argument("--max_attempts", type=int, default=5000,
                       help="Max sampling attempts before giving up")
    test_parser.add_argument("--output_vis", type=str, default="transformer_test_vis.png",
                       help="Filename for the final composite visualization")
    test_parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                       choices=["cpu","cuda"], help="Device for inference")

    args = parser.parse_args()

    if args.mode == "train":
        train_loop(args)
    elif args.mode == "test":
        test_loop(args)
    else:
        raise ValueError("Unknown mode. Use 'train' or 'test'.")
