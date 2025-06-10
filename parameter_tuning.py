#!/usr/bin/env python3
"""
Optuna-based parameter tuning for Tacotron2 using 1% of the LibriTTS dataset.
"""
import os
import sys
import json
import random

import torch
torch.manual_seed(42)
import numpy as np
import optuna
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# Add Tacotron2 model to path
BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/tacotron2",
    )
)
sys.path.append(BASE_DIR)
from model import Tacotron2

# Paths and constants
data_dir = "data/LibriTTS"
metadata_path = os.path.join(data_dir, "metadata.csv")
hparams_path = "hparams.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base (architecture) hyperparameters
def load_base_hparams():
    with open(hparams_path, "r") as f:
        return json.load(f)
BASE_HPARAMS = load_base_hparams()

# Dataset and collate
class TTSDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path):
        self.entries = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 3:
                    continue
                wav, text, spk = parts[:3]
                self.entries.append((wav, text.lower(), spk))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    def collate_fn(self, batch):
        wavs, texts, spks = zip(*batch)
        # text -> sequence
        seqs, ilens = tts_utils.prepare_input_sequence(list(texts))
        # ensure CPU long tensors
        seqs = seqs.cpu().long() if isinstance(seqs, torch.Tensor) else torch.LongTensor(seqs)
        ilens = ilens.cpu().long() if isinstance(ilens, torch.Tensor) else torch.LongTensor(ilens)
        return wavs, seqs, ilens, spks

# Speaker embedding and mel extraction
tts_utils = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils"
)

def get_speaker_embedding(wav_path, classifier, cache_dir="embeddings"):
    os.makedirs(cache_dir, exist_ok=True)
    emb_path = os.path.join(cache_dir, os.path.basename(wav_path) + ".npy")
    if os.path.exists(emb_path):
        return torch.from_numpy(np.load(emb_path)).float().to(DEVICE)
    wav, sr = torchaudio.load(wav_path)
    emb = classifier.encode_batch(wav.to(DEVICE)).detach().cpu().squeeze(0).mean(dim=0).numpy()
    np.save(emb_path, emb)
    return torch.from_numpy(emb).float().to(DEVICE)

mel_transform = None

def build_mel_transform(hparams):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=hparams.get("sample_rate", 22050),
        n_fft=hparams.get("n_fft", 1024),
        win_length=hparams.get("win_length", 1024),
        hop_length=hparams.get("hop_length", 256),
        n_mels=hparams["n_mel_channels"],
        f_min=0.0,
        f_max=hparams.get("f_max", 8000)
    ).to(DEVICE)

def get_mel(wav_path, hparams):
    global mel_transform
    if mel_transform is None:
        mel_transform = build_mel_transform(hparams)
    wav, sr = torchaudio.load(wav_path)
    if sr != hparams.get("sample_rate", 22050):
        wav = torchaudio.functional.resample(wav, sr, hparams.get("sample_rate", 22050))
    mel = mel_transform(wav.to(DEVICE)).squeeze(0)
    return mel

# Loss functions
class Tacotron2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        mel_out, mel_out_post, gate_out, _ = outputs
        mel_tgt, gate_tgt = targets
        gate_tgt = gate_tgt.view(-1,1)
        gate_out = gate_out.view(-1,1)
        mel_loss = self.mse(mel_out, mel_tgt) + self.mse(mel_out_post, mel_tgt)
        gate_loss = self.bce(gate_out, gate_tgt)
        return mel_loss + gate_loss

class GuidedAttentionLoss(nn.Module):
    """Guided attention loss function module."""

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """
        att_ws: (B, T_max_out, T_max_in)
        ilens: (B,) LongTensor
        olens: (B,) LongTensor
        """
        device = att_ws.device  # use same device as attention weights
        ilens_list = ilens.tolist() if isinstance(ilens, torch.Tensor) else ilens
        olens_list = olens.tolist() if isinstance(olens, torch.Tensor) else olens

        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens_list, olens_list, device=device)
        if self.masks is None:
            self.masks = self._make_masks(ilens_list, olens_list, device=device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens, device='cpu'):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen), device=device)
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma, device=device
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma, device='cpu'):
        # PyTorch >= 1.10 recommends specifying 'indexing'
        try:
            grid_x, grid_y = torch.meshgrid(
                torch.arange(olen, device=device),
                torch.arange(ilen, device=device),
                indexing="ij"
            )
        except TypeError:
            grid_x, grid_y = torch.meshgrid(
                torch.arange(olen, device=device),
                torch.arange(ilen, device=device)
            )
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2))
        )

    def _make_masks(self, ilens, olens, device='cpu'):
        in_masks = self.make_non_pad_mask(ilens, device=device)  # (B, T_in)
        out_masks = self.make_non_pad_mask(olens, device=device)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)

    def make_non_pad_mask(self, lengths, device='cpu', xs=None, length_dim=-1):
        return ~self.make_pad_mask(lengths, device=device, xs=xs, length_dim=length_dim)

    def make_pad_mask(self, lengths, device='cpu', xs=None, length_dim=-1):
        if length_dim == 0:
            raise ValueError("length_dim cannot be 0: {}".format(length_dim))

        if not isinstance(lengths, list):
            lengths = lengths.tolist()
        bs = int(len(lengths))
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)

        seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=device)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand

        if xs is not None:
            assert xs.size(0) == bs, (xs.size(0), bs)

            if length_dim < 0:
                length_dim = xs.dim() + length_dim
            ind = tuple(
                slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
            )
            mask = mask[ind].expand_as(xs).to(xs.device)
        return mask

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-3):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.count = None, 0
        self.stop = False
    def __call__(self, val, model=None, path=None):
        if self.best is None or val < self.best - self.min_delta:
            self.best = val
            self.count = 0
            if path and model:
                torch.save(model.state_dict(), path)
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True

# Objective for Optuna
METADATA = metadata_path

def objective(trial):
    # Sample training hyperparams
    lr    = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    bs    = trial.suggest_categorical("batch_size", [4, 8, 16])
    gaw   = trial.suggest_float("guided_attn_loss_weight", 0.1, 1.0)
    sigma = trial.suggest_float("guided_attn_sigma", 0.1, 1.0)
    alpha = trial.suggest_float("guided_attn_alpha", 0.5, 2.0)

    # Prepare hyperparams
    hparams = BASE_HPARAMS.copy()
    hparams.update({
        "learning_rate": lr,
        "guided_attn_loss_weight": gaw,
        "guided_attn_sigma": sigma,
        "guided_attn_alpha": alpha,
    })

    # Build dataset subset (1%)
    full = TTSDataset(METADATA)
    n    = len(full)
    subn = max(1, int(n * 0.01))
    idx  = torch.randperm(n)[:subn]
    data = Subset(full, idx)

    # Split
    n_train = int(0.8 * subn)
    n_val   = int(0.1 * subn)
    n_test  = subn - n_train - n_val
    train_ds, val_ds, _ = random_split(
        data, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=full.collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=full.collate_fn)

    # Speaker encoder
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="./spkrec-checkpoint",
        run_opts={"device": DEVICE}
    )

    # Model, optimizer, losses
    model = Tacotron2(
        mask_padding=hparams["mask_padding"],
        n_mel_channels=hparams["n_mel_channels"],
        n_symbols=hparams["n_symbols"],
        symbols_embedding_dim=hparams["symbols_embedding_dim"],
        encoder_kernel_size=hparams["encoder_kernel_size"],
        encoder_n_convolutions=hparams["encoder_n_convolutions"],
        encoder_embedding_dim=hparams["encoder_embedding_dim"],
        attention_rnn_dim=hparams["attention_rnn_dim"],
        attention_dim=hparams["attention_dim"],
        attention_location_n_filters=hparams["attention_location_n_filters"],
        attention_location_kernel_size=hparams["attention_location_kernel_size"],
        n_frames_per_step=hparams["n_frames_per_step"],
        decoder_rnn_dim=hparams["decoder_rnn_dim"],
        prenet_dim=hparams["prenet_dim"],
        max_decoder_steps=hparams["max_decoder_steps"],
        gate_threshold=hparams["gate_threshold"],
        p_attention_dropout=hparams["p_attention_dropout"],
        p_decoder_dropout=hparams["p_decoder_dropout"],
        postnet_embedding_dim=hparams["postnet_embedding_dim"],
        postnet_kernel_size=hparams["postnet_kernel_size"],
        postnet_n_convolutions=hparams["postnet_n_convolutions"],
        decoder_no_early_stopping=hparams["decoder_no_early_stopping"],
        speaker_embedding_dim=hparams["speaker_embedding_dim"]
    ).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    taco  = Tacotron2Loss().to(DEVICE)
    ga    = GuidedAttentionLoss(sigma=sigma, alpha=alpha).to(DEVICE)
    stopper = EarlyStopping(patience=5, min_delta=1e-3)

    # Training loop
    epochs = 10
    best_val = float('inf')
    for ep in range(epochs):
        # Train
        model.train()
        for wavs, seqs, ilens, spks in train_loader:
            seqs, ilens = seqs.to(DEVICE), ilens.to(DEVICE)
            # Prepare targets
            mels = [get_mel(w, hparams) for w in wavs]
            mlens = [m.shape[1] for m in mels]
            maxlen = max(mlens)
            mel_tgt = torch.stack([F.pad(m, (0, maxlen-m.shape[1])) for m in mels]).to(DEVICE)
            gate_tgt = (torch.arange(maxlen).unsqueeze(0) >= torch.LongTensor(mlens).unsqueeze(1)).float().view(-1,1).to(DEVICE)
            spk_embs = torch.stack([get_speaker_embedding(w, classifier) for w in wavs]).to(DEVICE)

            # Forward/backward
            opt.zero_grad()
            outs = model((seqs, ilens, mel_tgt, maxlen, torch.LongTensor(mlens).to(DEVICE)), speaker_embedding=spk_embs)
            if len(outs)==3:
                mel_out, gate_out, attn = outs
                mel_out_post = model.postnet(mel_out)
                outs = (mel_out, mel_out_post, gate_out, attn)
            loss = taco(outs, (mel_tgt, gate_tgt)) + gaw * ga(outs[3], ilens, torch.LongTensor(mlens).to(DEVICE))
            loss.backward()
            opt.step()

        # Validate
        model.eval()
        vals = []
        with torch.no_grad():
            for wavs, seqs, ilens, spks in val_loader:
                seqs, ilens = seqs.to(DEVICE), ilens.to(DEVICE)
                mels = [get_mel(w, hparams) for w in wavs]
                mlens = [m.shape[1] for m in mels]
                maxlen = max(mlens)
                mel_tgt = torch.stack([F.pad(m,(0,maxlen-m.shape[1])) for m in mels]).to(DEVICE)
                gate_tgt = (torch.arange(maxlen).unsqueeze(0) >= torch.LongTensor(mlens).unsqueeze(1)).float().view(-1,1).to(DEVICE)
                spk_embs = torch.stack([get_speaker_embedding(w, classifier) for w in wavs]).to(DEVICE)

                outs = model((seqs, ilens, mel_tgt, maxlen, torch.LongTensor(mlens).to(DEVICE)), speaker_embedding=spk_embs)
                if len(outs)==3:
                    mel_out, gate_out, attn = outs
                    mel_out_post = model.postnet(mel_out)
                    outs = (mel_out, mel_out_post, gate_out, attn)
                val_loss = taco(outs, (mel_tgt, gate_tgt)) + gaw * ga(outs[3], ilens, torch.LongTensor(mlens).to(DEVICE))
                vals.append(val_loss.item())

        avg_val = float(np.mean(vals))
        trial.report(avg_val, ep)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if avg_val < best_val:
            best_val = avg_val
        stopper(avg_val, model=model, path="best_model.pth")
        if stopper.stop:
            break

    return best_val

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, timeout=3600)
    print("Best trial:", study.best_trial.params)
