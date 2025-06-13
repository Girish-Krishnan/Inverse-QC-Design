#!/usr/bin/env python3
"""
visualize_train_samples_full_spectrum.py

Visualize N random circuits from the training set, showing each circuit diagram
alongside its full measurement probability spectrum (all 2^n states).

Usage:
    python visualize_train_samples_full_spectrum.py --file data/crypto_qc_train.json --n 5
"""

import argparse
import json
import random

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit

# ------------------------------------------------------------
# Reconstruct a QuantumCircuit from token sequence
# ------------------------------------------------------------
def reconstruct_circuit(entry):
    """
    Given a dataset entry with:
      - "num_qubits": int
      - "tokens": list of gate tokens
    Reconstruct a Qiskit QuantumCircuit applying those gates.
    Unknown or out‐of‐range qubit references are skipped.
    Bucketed rotations "rx(a)", "ry(a)", "rz(a)" use angle = (a/8)*2π.
    """
    n = entry["num_qubits"]
    tokens = entry["tokens"]
    qc = QuantumCircuit(n)

    for tok in tokens:
        parts = tok.split()
        op = parts[0]

        # Single-qubit H, X, Y, Z
        if op in {"h", "x", "y", "z"} and len(parts) == 2:
            try:
                q = int(parts[1][1:])
                if 0 <= q < n:
                    getattr(qc, op)(q)
            except:
                continue

        # CNOT
        elif op == "cx" and len(parts) == 3:
            try:
                q1 = int(parts[1][1:])
                q2 = int(parts[2][1:])
                if 0 <= q1 < n and 0 <= q2 < n and q1 != q2:
                    qc.cx(q1, q2)
            except:
                continue

        # Bucketed rotations: "rx(a)", "ry(a)", "rz(a)"
        elif op.startswith("rx(") or op.startswith("ry(") or op.startswith("rz("):
            gate_name = op[:2]  # "rx","ry","rz"
            try:
                bucket = int(op[3:-1])
                angle = bucket * 2 * np.pi / 8
                q = int(parts[1][1:])
                if 0 <= q < n:
                    getattr(qc, gate_name)(angle, q)
            except:
                continue

        # E91 entangle: "e91_entangle q{i} q{j}"
        elif op == "e91_entangle" and len(parts) == 3:
            try:
                q1 = int(parts[1][1:])
                q2 = int(parts[2][1:])
                if 0 <= q1 < n and 0 <= q2 < n:
                    qc.h(q1)
                    qc.cx(q1, q2)
            except:
                continue

        # GHZ "ghz_entangle q0 q{i}"
        elif op == "ghz_entangle" and len(parts) == 3:
            try:
                q1 = int(parts[1][1:])
                q2 = int(parts[2][1:])
                if q1 == 0 and 0 <= q2 < n:
                    qc.cx(0, q2)
            except:
                continue

        # GHZ initial step "ghz_step q0"
        elif op == "ghz_step" and len(parts) == 2:
            try:
                q = int(parts[1][1:])
                if q == 0 and 0 <= q < n:
                    qc.h(0)
            except:
                continue

        # BB84 is a placeholder; skip
        elif op == "bb84_step":
            continue

        else:
            # Ignore unrecognized token
            continue

    return qc

# ------------------------------------------------------------
# Plot full probability spectrum
# ------------------------------------------------------------
def plot_full_spectrum(prob_dict, nq, ax):
    """
    Plot all 2^n states as a bar chart, in lexicographic order.
    - prob_dict: {bitstring: probability}
    - nq: number of qubits
    - ax: matplotlib axis to draw on
    """
    num_states = 2 ** nq
    states = [format(i, f"0{nq}b") for i in range(num_states)]
    probs = [prob_dict.get(s, 0.0) for s in states]

    x = np.arange(num_states)
    ax.bar(x, probs, color="teal", edgecolor="black", width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=90, fontsize=6)
    ax.set_ylabel("Probability")
    ax.set_title(f"Full Measurement Spectrum (2^{nq} = {num_states} states)")
    ax.set_xlim(-0.5, num_states - 0.5)

# ------------------------------------------------------------
# Main visualization loop
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="data/crypto_qc_train.json",
        help="Path to train JSON file",
    )
    parser.add_argument(
        "--n", type=int, default=5, help="Number of random samples to visualize"
    )
    args = parser.parse_args()

    # Load dataset
    try:
        with open(args.file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File '{args.file}' not found.")
        return

    total = len(data)
    if total == 0:
        print("Dataset is empty.")
        return

    print(f"Loaded {total} entries from '{args.file}'")
    n_show = min(args.n, total)
    print(f"Visualizing {n_show} random samples…\n")

    # Randomly sample indices
    indices = random.sample(range(total), n_show)

    for idx in indices:
        entry = data[idx]
        structure = entry.get("structure", "N/A")
        nq = entry["num_qubits"]
        entropy_val = entry.get("entropy", None)
        ent_flag = entry.get("is_entangled", False)

        # Reconstruct circuit
        qc = reconstruct_circuit(entry)

        # Prepare figure with two subplots side by side
        fig, (ax_circ, ax_prob) = plt.subplots(
            1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1]}
        )

        # Draw the circuit on ax_circ
        try:
            qc.draw(output="mpl", ax=ax_circ)
        except Exception:
            ax_circ.text(
                0.5,
                0.5,
                "Circuit diagram unavailable",
                horizontalalignment="center",
                verticalalignment="center",
            )
        ax_circ.set_title(f"Circuit ({nq} qubits)")
        ax_circ.axis("off")

        # Plot the full probability spectrum on ax_prob
        plot_full_spectrum(entry["probs"], nq, ax_prob)

        # Overall title
        title = (
            f"Structure: {structure}  |  Qubits: {nq}  |  "
            f"Entropy: {entropy_val:.3f}  |  Entangled: {'Yes' if ent_flag else 'No'}"
        )
        fig.suptitle(title, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()

if __name__ == "__main__":
    main()
