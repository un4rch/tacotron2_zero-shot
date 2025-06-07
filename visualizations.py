import os
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curves(train_losses, val_losses, out_path):
    """Plot train/val loss curves in same plot."""
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Tacotron2 Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_mel_spectrograms(mel_true, mel_pred, out_path, sr=22050, hop_length=256):
    """
    Plot original and predicted MEL spectrograms side by side.
    mel_true, mel_pred: (n_mel, T)
    """
    # Convert to dB
    import librosa
    mel_true_db = librosa.power_to_db(np.maximum(mel_true, 1e-10), ref=np.max)
    mel_pred_db = librosa.power_to_db(np.maximum(mel_pred, 1e-10), ref=np.max)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    im0 = axs[0].imshow(mel_true_db, aspect='auto', origin='lower')
    axs[0].set_title("Ground Truth MEL (dB)")
    plt.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(mel_pred_db, aspect='auto', origin='lower')
    axs[1].set_title("Predicted MEL (dB)")
    plt.colorbar(im1, ax=axs[1])
    for ax in axs:
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel Channels")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_gate_outputs(gate_outputs, gate_targets, threshold, out_path):
    """
    Plotea la salida del gate (sigmoid) para la primera muestra del batch.
    - gate_outputs: (T,) numpy array o torch 1D tensor de sigmoides del modelo (después de sigmoid)
    - gate_targets: (T,) numpy array o torch 1D tensor de targets (0/1)
    """
    if hasattr(gate_outputs, "cpu"):
        gate_outputs = gate_outputs.detach().cpu().numpy()
    if hasattr(gate_targets, "cpu"):
        gate_targets = gate_targets.detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(gate_outputs, label="Gate output (sigmoid)", lw=2)
    plt.plot(gate_targets, label="Gate target", lw=2, linestyle="dashed", alpha=0.6)
    plt.axhline(y=threshold, color='r', linestyle='--', label='Gate Threshold')
    plt.xlabel("Frame")
    plt.ylabel("Gate value")
    plt.title("Gate output vs Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
