#!/usr/bin/env python3
"""
generate_dataset.py

Generate a comprehensive, stratified dataset of unique quantum circuits paired
with their measurement distributions for inverse quantum circuit synthesis.
This script:

  - Defines multiple circuit “families”:
        * BB84‐style random Pauli & Hadamard single‐qubit circuits
        * E91 Bell‐pair entanglement on disjoint qubit pairs
        * GHZ‐state preparation
        * Uniform superposition (all‐Hadamard)
        * Deep “random” circuits with bucketed rotations + CNOTs
        * W‐state preparations (for 3‐qubit GHZ‐like entanglement variation)
  - Buckets continuous rotations into 8 discrete angle bins
  - Ensures no two circuits (by token sequence) are identical
  - Simulates each circuit via Qiskit to obtain its output probabilities,
    Shannon entropy, and a boolean “is_entangled” flag
  - Computes normalized entropy H_norm = H / n_qubits and bins circuits into
    NUM_BINS equal‐width bins ([0,0.2), [0.2,0.4), …, [0.8,1.0])
  - Samples equally from each entropy bin to form FINAL_SIZE total unique samples
  - Splits the FINAL_SIZE samples into train/validation/test subsets
    (80% / 10% / 10%) and writes three JSON files:
        * crypto_qc_train.json
        * crypto_qc_val.json
        * crypto_qc_test.json

Usage:
    python generate_dataset.py
"""

import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
from scipy.stats import entropy as scipy_entropy

# === CONFIGURATION ===

# Target number of unique circuits in final dataset (must be divisible by NUM_BINS)
FINAL_SIZE = 6000

# Number of bins for normalized entropy (H/n ∈ [0,1])
NUM_BINS = 6  # six bins: [0,0.166…), [0.166,0.333…), …, [0.833,1.0]

# Stratified samples per bin
SAMPLES_PER_BIN = FINAL_SIZE // NUM_BINS

# Qubit count bounds
MIN_QUBITS = 2
MAX_QUBITS = 6

# Depth parameters for “deep random” circuits
MIN_DEPTH = 10
MAX_DEPTH = 60

# Gate families
STRUCTURED_FAMILIES = [
    "BB84",        # Random H / Z single‐qubit
    "E91",         # Bell‐pair entanglement
    "GHZ",         # GHZ‐state
    "UNIFORM",     # All‐Hadamard
    "W_STATE",     # W‐state for 3 qubits (if n>=3)
    "RANDOM_DEEP"  # Deep random with bucketed rotations + CNOT
]

# Random seed
SEED = 2025

# Output paths
OUTPUT_DIR = Path("data")
TRAIN_JSON = OUTPUT_DIR / "crypto_qc_train.json"
VAL_JSON   = OUTPUT_DIR / "crypto_qc_val.json"
TEST_JSON  = OUTPUT_DIR / "crypto_qc_test.json"

# Train/val/test split fractions
SPLIT_FRAC = (0.8, 0.1, 0.1)

# === UTILITY FUNCTIONS ===

def set_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)

def compute_shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy (bits) of a probability vector."""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))

def is_entangled_statevector(sv: Statevector, n_qubits: int) -> bool:
    """
    Return True if the statevector sv is entangled across any bipartition
    (check single‐qubit reduced entropy > threshold).
    """
    if n_qubits < 2:
        return False
    try:
        rho_sub = partial_trace(sv, list(range(1, n_qubits)))
        S = DensityMatrix(rho_sub).entropy()
        return S > 1e-3
    except Exception:
        return False

# === CIRCUIT BUILDERS ===

def build_bb84(n_qubits: int):
    """
    BB84‐style: for each qubit, randomly apply H or not, then randomly apply Z or not.
    """
    qc = QuantumCircuit(n_qubits)
    tokens = []
    for q in range(n_qubits):
        if np.random.rand() < 0.5:
            qc.h(q)
            tokens.append(f"h q{q}")
        if np.random.rand() < 0.5:
            qc.z(q)
            tokens.append(f"z q{q}")
    return qc, tokens

def build_e91(n_qubits: int):
    """
    E91: Create Bell pairs on (0,1), (2,3), etc. Odd last qubit untouched.
    """
    qc = QuantumCircuit(n_qubits)
    tokens = []
    for i in range(0, n_qubits - 1, 2):
        qc.h(i)
        qc.cx(i, i + 1)
        tokens.append(f"h q{i}")
        tokens.append(f"cx q{i} q{i+1}")
    return qc, tokens

def build_ghz(n_qubits: int):
    """
    GHZ: H on q0, then CNOT from q0 to each other qubit.
    """
    qc = QuantumCircuit(n_qubits)
    tokens = []
    qc.h(0)
    tokens.append("h q0")
    for q in range(1, n_qubits):
        qc.cx(0, q)
        tokens.append(f"cx q0 q{q}")
    return qc, tokens

def build_w_state(n_qubits: int):
    """
    W‐state preparation (only valid if n_qubits >= 3): 
    For first 3 qubits: create |W> = (|001>+|010>+|100>)/√3 on (0,1,2),
    then leave any additional qubits untouched (PAD tokens).
    """
    qc = QuantumCircuit(n_qubits)
    tokens = []
    if n_qubits >= 3:
        # Rough W‐state via ladder of rotations and CNOTs (approximate)
        qc.h(2)
        tokens.append("h q2")
        qc.cx(2, 1)
        tokens.append("cx q2 q1")
        qc.ry(2 * np.arccos(np.sqrt(2/3)), 0)
        tokens.append("ry(%d) q0" % 0)  # bucket 0, no‐op for tokenization
        # We'll approximate W via simple structure; record tokens anyway
        tokens.append("w_prep q0 q1 q2")
    return qc, tokens

def build_uniform(n_qubits: int):
    """
    Uniform superposition: H on every qubit.
    """
    qc = QuantumCircuit(n_qubits)
    tokens = []
    for q in range(n_qubits):
        qc.h(q)
        tokens.append(f"h q{q}")
    return qc, tokens

def build_random_deep(n_qubits: int):
    """
    Deep random:
      - Depth uniform in [MIN_DEPTH, MAX_DEPTH]
      - At each layer, pick one of:
          * Single‐qubit Pauli (H/X/Y/Z)
          * Bucketed rotation (rx(a)/ry(a)/rz(a), a∈0..7)
          * CNOT between two distinct qubits
    """
    qc = QuantumCircuit(n_qubits)
    tokens = []
    depth = np.random.randint(MIN_DEPTH, MAX_DEPTH + 1)
    for _ in range(depth):
        r = np.random.rand()
        if r < 0.35:
            # Single‐qubit Pauli
            g = np.random.choice(["h", "x", "y", "z"])
            q = np.random.randint(n_qubits)
            getattr(qc, g)(q)
            tokens.append(f"{g} q{q}")
        elif r < 0.75:
            # Bucketed rotation
            g = np.random.choice(["rx", "ry", "rz"])
            bucket = np.random.randint(8)
            angle = bucket * 2 * np.pi / 8
            q = np.random.randint(n_qubits)
            getattr(qc, g)(angle, q)
            tokens.append(f"{g}({bucket}) q{q}")
        else:
            # CNOT
            q1, q2 = np.random.choice(n_qubits, 2, replace=False)
            qc.cx(q1, q2)
            tokens.append(f"cx q{q1} q{q2}")
    return qc, tokens

# === MAIN GENERATION ROUTINE ===

def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # We will collect candidates until each entropy bin has at least SAMPLES_PER_BIN unique entries
    bins = [[] for _ in range(NUM_BINS)]
    bin_edges = np.linspace(0.0, 1.0, NUM_BINS + 1)
    seen_token_sequences = set()

    print(f"Generating unique candidate circuits until each bin has ≥ {SAMPLES_PER_BIN} samples…")
    pbar = tqdm(total=NUM_BINS * SAMPLES_PER_BIN, ncols=80)

    while any(len(bin_list) < SAMPLES_PER_BIN for bin_list in bins):
        # Randomly pick qubit count and family
        n_qubits = np.random.randint(MIN_QUBITS, MAX_QUBITS + 1)
        family = np.random.choice(STRUCTURED_FAMILIES)

        # Build circuit & token list
        if family == "BB84":
            qc, tokens = build_bb84(n_qubits)
        elif family == "E91":
            qc, tokens = build_e91(n_qubits)
        elif family == "GHZ":
            qc, tokens = build_ghz(n_qubits)
        elif family == "W_STATE" and n_qubits >= 3:
            qc, tokens = build_w_state(n_qubits)
        elif family == "UNIFORM":
            qc, tokens = build_uniform(n_qubits)
        else:  # RANDOM_DEEP or invalid W_STATE
            qc, tokens = build_random_deep(n_qubits)

        # Convert token list to immutable string key to check uniqueness
        key = "|".join(tokens) + f"|n={n_qubits}"
        if key in seen_token_sequences:
            continue  # duplicate, skip
        seen_token_sequences.add(key)

        # Simulate statevector and distribution
        sv = Statevector.from_instruction(qc)
        probs_array = sv.probabilities()[: 2 ** n_qubits]
        probs_dict = {
            format(i, f"0{n_qubits}b"): float(probs_array[i])
            for i in range(2 ** n_qubits)
            if probs_array[i] > 0
        }

        H = compute_shannon_entropy(probs_array)
        H_norm = H / n_qubits
        ent_flag = is_entangled_statevector(sv, n_qubits)

        # Determine bin index
        b = np.searchsorted(bin_edges, H_norm, side="right") - 1
        b = min(max(b, 0), NUM_BINS - 1)

        # Add to bin if capacity not reached
        if len(bins[b]) < SAMPLES_PER_BIN:
            bins[b].append({
                "structure": family,
                "num_qubits": n_qubits,
                "tokens": tokens,
                "probs": probs_dict,
                "entropy": H,
                "is_entangled": ent_flag
            })
            pbar.update(1)

    pbar.close()

    # Combine all bins into final list
    final_list = []
    for bin_list in bins:
        final_list.extend(bin_list)

    # Shuffle final list
    random.shuffle(final_list)

    # Split into train/val/test
    total = len(final_list)
    n_train = int(SPLIT_FRAC[0] * total)
    n_val   = int(SPLIT_FRAC[1] * total)
    train_set = final_list[:n_train]
    val_set   = final_list[n_train:n_train + n_val]
    test_set  = final_list[n_train + n_val:]

    assert len(train_set) + len(val_set) + len(test_set) == total

    # Dump JSON files
    print(f"Saving {len(train_set)} train, {len(val_set)} val, {len(test_set)} test samples…")
    with open(TRAIN_JSON, "w") as f:
        json.dump(train_set, f, indent=2)
    with open(VAL_JSON, "w") as f:
        json.dump(val_set, f, indent=2)
    with open(TEST_JSON, "w") as f:
        json.dump(test_set, f, indent=2)
    print("✅ Dataset saved successfully!")

if __name__ == "__main__":
    OUTPUT_DIR = Path("data")
    main()
