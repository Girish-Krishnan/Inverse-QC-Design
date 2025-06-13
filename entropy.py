import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_entropy(p):
    p = np.array(p)
    p = p[p > 0]  # Avoid log(0)
    return -np.sum(p * np.log2(p))

# === Load dataset ===
with open("data/crypto_qc_train.json", "r") as f:
    data = json.load(f)

entropies = []

for entry in tqdm(data):
    probs_dict = entry["probs"]
    num_qubits = entry["num_qubits"]
    p = np.zeros(2**num_qubits)
    for i in range(2**num_qubits):
        bitstring = format(i, f"0{num_qubits}b")
        p[i] = probs_dict.get(bitstring, 0.0)
    entropies.append(compute_entropy(p))

# === Plot histogram ===
plt.figure(figsize=(6, 4))
plt.hist(entropies, bins=30, color="slateblue", edgecolor="black")
plt.title("Entropy Distribution of Output Distributions")
plt.xlabel("Shannon Entropy (bits)")
plt.ylabel("Number of Samples")
plt.grid(True)
plt.tight_layout()
plt.savefig("entropy_distribution.png", dpi=300)
print("✅ Saved: entropy_distribution.png")
