import json
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Load data
# -----------------------
with open("results/diffusion/training_logs.json") as f:
    diff_logs = json.load(f)
with open("results/graphvae/training_logs.json") as f:
    vae_logs = json.load(f)
with open("results/transformer/training_logs.json") as f:
    tf_logs = json.load(f)

epochs = diff_logs["epoch"]

# -----------------------
# Line Plot: Validation Loss Comparison
# -----------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, diff_logs["val_CE"], label="Diffusion", linewidth=2)
plt.plot(epochs, vae_logs["val_loss"], label="GraphVAE", linewidth=2)
plt.plot(epochs, tf_logs["val_xent"], label="Transformer", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("val_loss_curve.png", dpi=300)

# -----------------------
# Bar Chart: Final Validation Loss Comparison
# -----------------------
final_losses = [
    diff_logs["val_CE"][-1],
    vae_logs["val_loss"][-1],
    tf_logs["val_xent"][-1]
]
labels = ["Diffusion", "GraphVAE", "Transformer"]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, final_losses, color=["#4c72b0", "#55a868", "#c44e52"])
plt.ylabel("Final Validation Loss")
plt.title("Final Validation Loss Comparison")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig("final_val_loss_bar.png", dpi=300)

# -----------------------
# Grid: Train vs Val Loss for Each Model
# -----------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
axs[0].plot(epochs, diff_logs["train_CE"], label="Train")
axs[0].plot(epochs, diff_logs["val_CE"], label="Val")
axs[0].set_title("Diffusion")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(epochs, vae_logs["train_loss"], label="Train")
axs[1].plot(epochs, vae_logs["val_loss"], label="Val")
axs[1].set_title("GraphVAE")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(epochs, tf_logs["train_xent"], label="Train")
axs[2].plot(epochs, tf_logs["val_xent"], label="Val")
axs[2].set_title("Transformer")
axs[2].legend()
axs[2].grid(True)

for ax in axs:
    ax.set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
plt.suptitle("Train vs Validation Loss for Each Model")
plt.tight_layout()
plt.savefig("train_vs_val_loss.png", dpi=300)

# -----------------------
# KL Divergence Comparison (TF and GraphVAE)
# -----------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, tf_logs["train_kl"], label="Transformer (KL)", linestyle="--")
plt.plot(epochs, vae_logs["train_loss"], label="GraphVAE (Total Loss)", linestyle=":")
plt.ylabel("KL Divergence / Total Loss")
plt.xlabel("Epoch")
plt.title("KL Divergence Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("kl_comparison.png", dpi=300)

print("All plots saved.")
