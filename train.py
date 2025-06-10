import os
import sys
import json
import torch
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import torch
import torch.distributed as dist           # ← y ya puedes usar dist.init_process_group(...)
from torch.nn.parallel import DistributedDataParallel as DDP
import torchaudio
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset, DistributedSampler
from torch import nn, optim
from speechbrain.inference.speaker import EncoderClassifier
from visualizations import plot_loss_curves, plot_mel_spectrograms, plot_gate_outputs
import mlflow
import mlflow.pytorch
import pandas as pd
import librosa
from pesq import pesq
from pystoi import stoi
import pyworld as pw
from scipy.spatial.distance import euclidean

# === Path Setup ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "DeepLearningExamples", "PyTorch", "SpeechSynthesis", "Tacotron2", "tacotron2"))
sys.path.append(BASE_DIR)
from model import Tacotron2

tts_utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils")

CHECKPOINTS_DIR = "checkpoints"
EMBEDDINGS_DIR = "embeddings"
VIS_DIR = "visualizations"
DATA_DIR = "data/LibriTTS"
METADATA = os.path.join(DATA_DIR, "metadata.csv")
HPARAMS_PATH = "hparams.json"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# === Download Dataset if Needed ===
"""if not os.path.exists("data/LibriTTS"):
    os.makedirs("data/LibriTTS", exist_ok=True)
    print("Downloading LibriTTS...")
    from torchaudio.datasets import LIBRISPEECH
    _ = LIBRISPEECH(root='data/LibriTTS', download=True)"""

# === Hyperparameters ===
with open(HPARAMS_PATH, "r") as f:
    hparams = json.load(f)

# === DistributedDataParallel setup ===
dist.init_process_group(backend="nccl")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# === MLflow Setup (solo rank 0) ===
if dist.get_rank() == 0:
    MLFLOW_TRACKING_URI = "http://admin:mlflow_password@mlflow.172.16.57.20.nip.io/"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Tacotron2_Zero-Shot")
    mlflow_run = mlflow.start_run(run_name=f"run_{os.getpid()}")
    # Guarda hiperparámetros SOLO EN RANK 0
    mlflow.log_params(hparams)
    mlflow.log_param("device", str(device))
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("weight_decay", 1e-5)
    mlflow.log_param("guided_attn_loss_weight", 0.2)

# === Load Speaker Encoder (ahora que DDP está iniciado) ===
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",  # explícita para que use un directorio fijo
    run_opts={"device": device}
)
# Sincronizamos para que rank>0 espere a que rank0 termine la descarga
dist.barrier()

# === Dataset Utilities ===
class TTSDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path):
        self.entries = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 3:
                    continue
                wav_path, transcript, speaker_id = parts[:3]
                self.entries.append({
                    "wav_path": wav_path,
                    "transcript": transcript,
                    "speaker_id": speaker_id
                })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        return entry["wav_path"], entry["transcript"], entry["speaker_id"]

    def collate_fn(self, batch):
        wav_paths, transcripts, speaker_ids = zip(*batch)
        text_input = [t.lower() for t in transcripts]
        sequences, input_lengths = tts_utils.prepare_input_sequence(text_input)
        if torch.is_tensor(sequences):
            sequences = sequences.cpu().numpy()
        if torch.is_tensor(input_lengths):
            input_lengths = input_lengths.cpu().numpy()
        text_padded = torch.LongTensor(sequences)
        input_lengths = torch.LongTensor(input_lengths)
        return wav_paths, text_padded, input_lengths, speaker_ids


# === Speaker Embedding Caching ===
def get_speaker_embedding(wav_path, classifier, embeddings_dir=EMBEDDINGS_DIR):
    emb_path = os.path.join(embeddings_dir, os.path.basename(wav_path) + ".npy")
    if os.path.exists(emb_path):
        emb = np.load(emb_path)
        return torch.from_numpy(emb).float().to(device)
    waveform, sr = torchaudio.load(wav_path)
    embedding = classifier.encode_batch(waveform.to(device)).detach().cpu().squeeze(0).mean(dim=0).numpy()
    np.save(emb_path, embedding)
    return torch.from_numpy(embedding).float().to(device)

# === MEL extraction transform ===
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=hparams.get("sample_rate", 22050),
    n_fft=hparams.get("n_fft", 1024),
    win_length=hparams.get("win_length", 1024),
    hop_length=hparams.get("hop_length", 256),
    n_mels=hparams["n_mel_channels"],
    f_min=0.0,
    f_max=hparams.get("f_max", 8000)
).to(device)

def get_mel(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    if sr != hparams.get("sample_rate", 22050):
        waveform = torchaudio.functional.resample(waveform, sr, hparams.get("sample_rate", 22050))
    mel = mel_transform(waveform.to(device))
    mel = mel.squeeze(0)  # [n_mel, T]
    return mel

def get_gate_target(mel, stop_frames=1):
    T = mel.shape[1]
    gate = torch.zeros(T, device=mel.device)
    gate[-stop_frames:] = 1.0
    return gate

# === Tacotron2Loss ===
class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = self.mse(mel_out, mel_target) + self.mse(mel_out_postnet, mel_target)
        gate_loss = self.bce(gate_out, gate_target)
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

class Tacotron2CombinedLoss(nn.Module):
    def __init__(self, guided_attn_loss_weight=0.2, sigma=0.4, alpha=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.guided_attn = GuidedAttentionLoss(sigma=sigma, alpha=alpha)
        self.guided_attn_loss_weight = guided_attn_loss_weight

    def forward(self, model_output, targets, input_lengths, output_lengths):
        mel_target, gate_target = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        mel_out, mel_out_postnet, gate_out, alignments = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = self.mse(mel_out, mel_target) + self.mse(mel_out_postnet, mel_target)
        gate_loss = self.bce(gate_out, gate_target)
        # Guided attention loss
        guided_loss = self.guided_attn(
            alignments, input_lengths, output_lengths
        )
        total_loss = mel_loss + gate_loss + self.guided_attn_loss_weight * guided_loss
        return total_loss

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, verbose=True, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model=None):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            if self.verbose and dist.get_rank() == 0:
                print(f"Validation loss improved: {self.best_loss} -> {val_loss}")
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path and model is not None:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.verbose and dist.get_rank() == 0:
                print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# === Prepare Dataset and Dataloaders (train/val/test) ===
full_dataset = TTSDataset(METADATA)
collate = full_dataset.collate_fn

# 1) Subset opcional
total_samples = len(full_dataset)
subset_frac   = hparams.get("subset_frac", 1.0)
seed          = hparams.get("seed", 42)
if subset_frac < 1.0:
    subset_len = int(total_samples * subset_frac)
    gen        = torch.Generator().manual_seed(seed)
    idx        = torch.randperm(total_samples, generator=gen)[:subset_len]
    dataset    = Subset(full_dataset, idx)
    if dist.get_rank() == 0:
        print(f"⚡ Usando sólo {subset_len}/{total_samples} ejemplos"
              f" ({subset_frac*100:.1f}% del dataset completo)")
else:
    dataset = full_dataset
    if dist.get_rank() == 0:
        print(f"✅ Usando el 100% del dataset ({total_samples} ejemplos)")

# 2) Split train/val/test
n = len(dataset)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
n_test  = n - n_train - n_val
train_set, val_set, test_set = random_split(
    dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(seed)
)
if dist.get_rank() == 0:
    print(f"→ Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}")

# 3) Dataloaders con DistributedSampler
use_workers = 0 if subset_frac < 1.0 else hparams.get("num_workers", 4)
if dist.get_rank() == 0:
    print(f"→ DataLoader workers: {use_workers}")

train_sampler = DistributedSampler(train_set,
                                   num_replicas=dist.get_world_size(),
                                   rank=dist.get_rank(),
                                   shuffle=True, seed=seed)
val_sampler   = DistributedSampler(val_set,
                                   num_replicas=dist.get_world_size(),
                                   rank=dist.get_rank(),
                                   shuffle=False, seed=seed)
test_sampler  = DistributedSampler(test_set,
                                   num_replicas=dist.get_world_size(),
                                   rank=dist.get_rank(),
                                   shuffle=False, seed=seed)

train_loader = DataLoader(train_set,
                          batch_size=hparams["batch_size"],
                          sampler=train_sampler,
                          collate_fn=collate,
                          num_workers=use_workers,
                          pin_memory=True)
val_loader   = DataLoader(val_set,
                          batch_size=hparams["batch_size"],
                          sampler=val_sampler,
                          collate_fn=collate,
                          num_workers=use_workers,
                          pin_memory=True)
test_loader  = DataLoader(test_set,
                          batch_size=1,        # 1 para inferencia
                          sampler=test_sampler,
                          collate_fn=collate,
                          num_workers=use_workers,
                          pin_memory=True)

# === Build Model ===
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
).to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=1e-5)
criterion = Tacotron2CombinedLoss(guided_attn_loss_weight=0.2, sigma=0.4, alpha=1.0)
# Early stopping
early_stopper = EarlyStopping(
    patience=hparams.get("early_stopping_patience", 20),
    min_delta=1e-3,
    verbose=True,
    save_path=os.path.join(CHECKPOINTS_DIR, "best_model.pth")
)

# Antes del bucle de entrenamiento, instanciamos las pérdidas “oficiales”
taco_loss_fn    = Tacotron2Loss().to(device)
guided_loss_fn  = GuidedAttentionLoss(
    sigma=hparams.get("guided_attn_sigma", 0.4),
    alpha=hparams.get("guided_attn_alpha", 1.0)
).to(device)
attn_weight     = hparams.get("guided_attn_loss_weight", 0.2)

train_loss_history = []
train_eval_history = []
val_loss_history = []

import matplotlib.pyplot as plt

if dist.get_rank() == 0:
    print(f"Starting training... {hparams['epochs']} epochs.")

for epoch in range(hparams['epochs']):
    train_sampler.set_epoch(epoch)
    model.train()
    epoch_train_losses = []
    gate_stats_train   = []

    # --- Training ---
    for wav_paths, text_padded, input_lengths, speaker_ids in train_loader:
        # === 1) Prepara MELs y targets ===
        mels = [get_mel(wp) for wp in wav_paths]
        mel_lengths = [m.shape[1] for m in mels]
        max_len     = max(mel_lengths)

        # a) mel_targets padded -> [B, n_mel, T]
        mel_targets = torch.stack([
            F.pad(m, (0, max_len - m.shape[1]))
            for m in mels
        ], dim=0).to(device)

        # b) gate_targets -> [B, T]
        arange      = torch.arange(max_len, device=device).unsqueeze(0)      # (1, T)
        lengths     = torch.LongTensor(mel_lengths).unsqueeze(1).to(device)  # (B,1)
        output_lengths = lengths.squeeze(1)                                  # (B,)
        gate_targets   = (arange >= lengths).float()                         # (B, T)

        # c) speaker embeddings -> [B, D]
        spk_embeddings = torch.stack([
            get_speaker_embedding(wp, classifier)
            for wp in wav_paths
        ], dim=0).to(device)

        # === 2) Forward pass ===
        inputs = (
            text_padded.to(device),
            input_lengths.to(device),
            mel_targets,
            max_len,
            output_lengths
        )
        model.zero_grad()
        outs = model(inputs, speaker_embedding=spk_embeddings)

        # unpack seguro (parche al vendor code)
        if len(outs) == 3:
            mel_out, gate_out, alignments = outs
            mel_out_postnet = model.module.postnet(mel_out) if isinstance(model, DDP) else model.postnet(mel_out)
        else:
            mel_out, mel_out_postnet, gate_out, alignments = outs

        # === 3) Pérdida Tacotron2 (mel + gate) ===
        mg_loss = taco_loss_fn(
            model_output=(mel_out, mel_out_postnet, gate_out, alignments),
            targets=(mel_targets, gate_targets)
        )

        # === 4) Guided Attention Loss ===
        ga_loss = guided_loss_fn(
            alignments,
            input_lengths.to(device),
            output_lengths
        )

        # === 5) Total y backward ===
        loss = mg_loss + attn_weight * ga_loss
        loss.backward()
        optimizer.step()
        epoch_train_losses.append(loss.item())

        # === 6) Estadísticas de gate para debug ===
        with torch.no_grad():
            gs = torch.sigmoid(gate_out).cpu()
            gate_stats_train.append([
                gs.mean().item(),
                gs.min().item(),
                gs.max().item()
            ])

    # --- Print resumen train ---  
    gate_stats_train = np.array(gate_stats_train)
    if dist.get_rank() == 0:
        print(f"[Train][Epoch {epoch}] Gate sigmoid mean: {gate_stats_train[:,0].mean():.4f} "
              f"| min: {gate_stats_train[:,1].min():.4f} "
              f"| max: {gate_stats_train[:,2].max():.4f}")

    avg_train_loss = np.mean(epoch_train_losses)
    train_loss_history.append(avg_train_loss)

    # === Train-eval loop (mismo train_loader pero en .eval()) ===
    model.eval()
    train_eval_losses = []
    with torch.no_grad():
        for wav_paths, text_padded, input_lengths, speaker_ids in train_loader:
            mels        = [get_mel(w) for w in wav_paths]
            mel_lengths = [m.shape[1] for m in mels]
            max_len     = max(mel_lengths)
            mel_tgt     = torch.stack([F.pad(m, (0, max_len-m.shape[1])) for m in mels], dim=0).to(device)

            arange      = torch.arange(max_len, device=device).unsqueeze(0)
            lengths     = torch.LongTensor(mel_lengths).unsqueeze(1).to(device)
            out_lens    = lengths.squeeze(1)
            gate_tgt    = (arange >= lengths).float()

            spk_emb     = torch.stack([get_speaker_embedding(w, classifier) for w in wav_paths], dim=0).to(device)
            inputs      = (text_padded.to(device), input_lengths.to(device), mel_tgt, max_len, out_lens)

            outs = model(inputs, speaker_embedding=spk_emb)
            if len(outs) == 3:
                mel_out, gate_out, alignments = outs
                mel_out_post = model.module.postnet(mel_out) if isinstance(model, DDP) else model.postnet(mel_out)
            else:
                mel_out, mel_out_post, gate_out, alignments = outs

            mg = taco_loss_fn(
                model_output=(mel_out, mel_out_post, gate_out, alignments),
                targets=(mel_tgt, gate_tgt)
            )
            ga = guided_loss_fn(alignments, input_lengths.to(device), out_lens)
            train_eval_losses.append((mg + attn_weight * ga).item())

    avg_train_eval = float(np.mean(train_eval_losses))
    train_eval_history.append(avg_train_eval)

    # === Validation loop ===
    epoch_val_losses = []
    gate_stats_val   = []
    val_sample       = None

    with torch.no_grad():
        for wav_paths, text_padded, input_lengths, speaker_ids in val_loader:
            mels        = [get_mel(wp) for wp in wav_paths]
            mel_lengths = [m.shape[1] for m in mels]
            max_len     = max(mel_lengths)

            mel_targets = torch.stack([
                F.pad(m, (0, max_len - m.shape[1])) for m in mels
            ], dim=0).to(device)

            arange        = torch.arange(max_len, device=device).unsqueeze(0)
            lengths       = torch.LongTensor(mel_lengths).unsqueeze(1).to(device)
            output_lengths = lengths.squeeze(1)
            gate_targets   = (arange >= lengths).float()

            spk_embeddings = torch.stack([
                get_speaker_embedding(wp, classifier) for wp in wav_paths
            ], dim=0).to(device)

            inputs = (
                text_padded.to(device),
                input_lengths.to(device),
                mel_targets,
                max_len,
                output_lengths
            )
            outs = model(inputs, speaker_embedding=spk_embeddings)
            if len(outs) == 3:
                mel_out, gate_out, alignments = outs
                mel_out_postnet = model.module.postnet(mel_out) if isinstance(model, DDP) else model.postnet(mel_out)
            else:
                mel_out, mel_out_postnet, gate_out, alignments = outs

            mg_loss = taco_loss_fn(
                model_output=(mel_out, mel_out_postnet, gate_out, alignments),
                targets=(mel_targets, gate_targets)
            )
            ga_loss = guided_loss_fn(alignments, input_lengths.to(device), output_lengths)
            val_loss = mg_loss + attn_weight * ga_loss
            epoch_val_losses.append(val_loss.item())

            gs = torch.sigmoid(gate_out).cpu()
            gate_stats_val.append([gs.mean().item(), gs.min().item(), gs.max().item()])

            if val_sample is None:
                val_sample = {
                    'mel_true': mel_targets[0].cpu().numpy(),
                    'mel_pred': mel_out_postnet[0].cpu().numpy(),
                    'gate_out': gs[0].numpy(),
                    'gate_tgt': gate_targets[0].cpu().numpy()
                }

    # --- Print resumen valid ---
    gate_stats_val = np.array(gate_stats_val)
    if dist.get_rank() == 0:
        print(f"[Valid][Epoch {epoch}] Gate sigmoid mean: {gate_stats_val[:,0].mean():.4f} "
              f"| min: {gate_stats_val[:,1].min():.4f} "
              f"| max: {gate_stats_val[:,2].max():.4f}")

    avg_val_loss = np.mean(epoch_val_losses)
    val_loss_history.append(avg_val_loss)

    # --- Ploteos y checkpoints ---
    if dist.get_rank() == 0:
        loss_curve_path = os.path.join(VIS_DIR, f"loss_curves_epoch{epoch}.png")
        plot_loss_curves(train_eval_history, val_loss_history, loss_curve_path)
        mlflow.log_artifact(loss_curve_path, artifact_path="plots")

        if epoch % hparams["checkpoint_interval"] == 0:
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINTS_DIR, f"tacotron2_epoch{epoch}.pth"))
            print(f"Saved checkpoint at epoch {epoch}")

            if val_sample is not None:
                mel_compare_path = os.path.join(VIS_DIR, f"mel_compare_epoch{epoch}.png")
                plot_mel_spectrograms(val_sample['mel_true'], val_sample['mel_pred'], mel_compare_path)
                mlflow.log_artifact(mel_compare_path, artifact_path="plots")
                gate_plot_path = os.path.join(VIS_DIR, f"gate_epoch{epoch}.png")
                plot_gate_outputs(val_sample['gate_out'], val_sample['gate_tgt'],
                                  threshold=hparams["gate_threshold"], out_path=gate_plot_path)
                mlflow.log_artifact(gate_plot_path, artifact_path="plots")

    mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
    mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

    if dist.get_rank() == 0:
        print(f"[Epoch {epoch}] "
              f"Train(train-mode mean): {train_loss_history[-1]:.1f}  |  "
              f"Train(eval-mode mean): {avg_train_eval:.1f}  |  "
              f"Val(eval-mode): {avg_val_loss:.1f}")

    # === EARLY STOPPING ===
    early_stopper(avg_val_loss, model)
    if early_stopper.early_stop:
        if dist.get_rank() == 0:
            print("Early stopping triggered.")
        best_model_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
        mlflow.log_artifact(best_model_path, artifact_path="models")
        break


if dist.get_rank() == 0:
    print("Training complete. Checkpoints saved in checkpoints/")
    mlflow.pytorch.log_model(model, artifact_path="final_model")

if dist.get_rank() == 0: # sintetiza todo el test set y calcula MCD, PESQ, STOI, F0‐RMSE y V/UV error

    # 1) Carga modelo y WaveGlow
    tacotron2 = Tacotron2(**hparams).to(device)
    tacotron2.load_state_dict(torch.load(
        os.path.join(CHECKPOINTS_DIR, "best_model.pth")
    ))
    tacotron2.eval()

    waveglow = torch.hub.load(
        "nvidia/DeepLearningExamples:torchhub",
        "nvidia_waveglow256pyt_fp16"
    )
    waveglow = waveglow.remove_weightnorm(waveglow).to(device).eval()

    report = []
    for wav_paths, text_padded, input_lengths, speaker_ids in test_loader:
        wp = wav_paths[0]
        # 2.1) GT audio
        gt_wav, sr = torchaudio.load(wp)
        gt = gt_wav.squeeze(0).numpy()
        # 2.2) Inferencia TTS
        with torch.no_grad():
            seq    = text_padded.to(device)
            ilen   = input_lengths.to(device)
            spk_emb= get_speaker_embedding(wp, classifier).unsqueeze(0)
            mel    = tacotron2.inference((seq, ilen),
                                         speaker_embedding=spk_emb)
            syn    = waveglow.infer(mel).squeeze().cpu().numpy()

        # 3) Métricas objetivas
        # 3.1 MCD
        gt_mfcc  = librosa.feature.mfcc(gt, sr=sr, n_mfcc=13)
        syn_mfcc = librosa.feature.mfcc(syn, sr=sr, n_mfcc=13)
        _, wp_d  = librosa.sequence.dtw(gt_mfcc.T, syn_mfcc.T,
                                        metric='euclidean')
        mcd = np.mean([
            euclidean(gt_mfcc[:,i], syn_mfcc[:,j]) for (i,j) in wp_d
        ]) * (10/np.log(10)*np.sqrt(2))

        # 3.2 PESQ (wideband)
        pesq_score = pesq(sr, gt, syn, 'wb')

        # 3.3 STOI
        stoi_score = stoi(gt, syn, sr, extended=False)

        # 3.4 F0‐RMSE & V/UV
        f0_gt, t_gt   = pw.dio(gt, sr);   f0_gt = pw.stonemask(gt, f0_gt, t_gt, sr)
        f0_syn, _    = pw.dio(syn, sr);  f0_syn= pw.stonemask(syn, f0_syn, t_gt, sr)
        f0_rmse      = float(np.sqrt(np.mean((f0_gt - f0_syn)**2)))
        vuv_err      = float(np.mean((f0_gt>0) != (f0_syn>0)))

        report.append({
            "wav": os.path.basename(wp),
            "MCD": float(mcd),
            "PESQ": float(pesq_score),
            "STOI": float(stoi_score),
            "F0_RMSE": f0_rmse,
            "VUV_error": vuv_err
        })

    # 4) Guarda CSV
    df      = pd.DataFrame(report)
    out_csv = os.path.join(CHECKPOINTS_DIR, "evaluation_report.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved evaluation report → {out_csv}")


dist.destroy_process_group()
