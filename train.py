import os
import sys
import json
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from speechbrain.inference.speaker import EncoderClassifier
from visualizations import plot_loss_curves, plot_mel_spectrograms, plot_gate_outputs
import mlflow
import mlflow.pytorch

# === MLflow Setup ===
MLFLOW_TRACKING_URI = "http://admin:mlflow_password@mlflow.172.16.57.20.nip.io/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Tacotron2_Zero-Shot")

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

device = "cuda" if torch.cuda.is_available() else "cpu"

# === MLflow Logging ===
with mlflow.start_run(run_name=f"run_{os.getpid()}") as run:
    # Guarda hiperparámetros
    mlflow.log_params(hparams)
    # (Opcional) Log de otros parámetros relevantes:
    mlflow.log_param("device", device)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("weight_decay", 1e-5)
    mlflow.log_param("guided_attn_loss_weight", 0.2)

# === Load Speaker Encoder ===
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

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
            if self.verbose:
                print(f"Validation loss improved: {self.best_loss} -> {val_loss}")
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path and model is not None:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# === Prepare Dataset and Dataloader ===
dataset = TTSDataset(METADATA)
n = len(dataset)
train_len = int(0.9 * n)
val_len = n - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_set, batch_size=hparams["batch_size"], shuffle=True, collate_fn=dataset.collate_fn)
val_loader = DataLoader(val_set, batch_size=hparams["batch_size"], shuffle=False, collate_fn=dataset.collate_fn)

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
optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=1e-5)
criterion = Tacotron2CombinedLoss(guided_attn_loss_weight=0.2, sigma=0.4, alpha=1.0)
# Early stopping
early_stopper = EarlyStopping(
    patience=hparams.get("early_stopping_patience", 20),
    min_delta=1e-3,
    verbose=True,
    save_path=os.path.join(CHECKPOINTS_DIR, "best_model.pth")
)

train_loss_history = []
val_loss_history = []

import matplotlib.pyplot as plt

print(f"Starting training... {hparams["epochs"]} epochs.")
for epoch in range(hparams["epochs"]):
    model.train()
    epoch_train_losses = []
    gate_stats_train = []

    for wav_paths, text_padded, input_lengths, speaker_ids in train_loader:
        # === Extract MELs and gates for the batch
        mels = [get_mel(wp) for wp in wav_paths]
        gates = [get_gate_target(m) for m in mels]
        mel_lengths = [m.shape[1] for m in mels]
        max_len = max(mel_lengths)
        mel_targets = torch.stack([F.pad(m, (0, max_len - m.shape[1])) for m in mels]).to(device)  # [B, n_mel, T]
        gate_targets = torch.stack([F.pad(g, (0, max_len - g.shape[0])) for g in gates]).to(device)  # [B, T]
        output_lengths = torch.LongTensor(mel_lengths).to(device)
        # === Speaker Embeddings for batch (zero-shot ready)
        spk_embeddings = torch.stack([get_speaker_embedding(wp, classifier) for wp in wav_paths])
        #print("spk_embeddings shape:", spk_embeddings.shape)  # Debe ser [B, 192]
        inputs = (text_padded.to(device), input_lengths.to(device), mel_targets, max_len, output_lengths)
        model.zero_grad()
        outputs = model(inputs, speaker_embedding=spk_embeddings)
        loss = criterion(
            outputs, (mel_targets, gate_targets), 
            input_lengths=input_lengths, 
            output_lengths=output_lengths
        )
        loss.backward()
        optimizer.step()

        epoch_train_losses.append(loss.item())

        # === DEBUG gate values for train ===
        with torch.no_grad():
            gate_out = outputs[2]   # [B, T] o [B*T, 1]
            gate_sigmoid = torch.sigmoid(gate_out).detach().cpu()
            gate_stats_train.append([
                gate_sigmoid.mean().item(),
                gate_sigmoid.min().item(),
                gate_sigmoid.max().item()
            ])

    # Print train gate stats summary
    gate_stats_train = np.array(gate_stats_train)
    print(f"[Train][Epoch {epoch}] Gate sigmoid mean: {gate_stats_train[:,0].mean():.4f} | min: {gate_stats_train[:,1].min():.4f} | max: {gate_stats_train[:,2].max():.4f}")

    avg_train_loss = np.mean(epoch_train_losses)
    train_loss_history.append(avg_train_loss)

    # === Validation loop
    model.eval()
    epoch_val_losses = []
    gate_stats_val = []
    val_sample = None
    with torch.no_grad():
        for i, (wav_paths, text_padded, input_lengths, speaker_ids) in enumerate(val_loader):
            mels = [get_mel(wp) for wp in wav_paths]
            gates = [get_gate_target(m) for m in mels]
            mel_lengths = [m.shape[1] for m in mels]
            max_len = max(mel_lengths)
            mel_targets = torch.stack([F.pad(m, (0, max_len - m.shape[1])) for m in mels])
            gate_targets = torch.stack([F.pad(g, (0, max_len - g.shape[0])) for g in gates])
            output_lengths = torch.LongTensor(mel_lengths).to(device)
            spk_embeddings = torch.stack([get_speaker_embedding(wp, classifier) for wp in wav_paths])

            inputs = (text_padded.to(device), input_lengths.to(device), mel_targets, max_len, output_lengths)
            outputs = model(inputs, speaker_embedding=spk_embeddings)
            val_loss = criterion(
                outputs, (mel_targets, gate_targets), 
                input_lengths=input_lengths, 
                output_lengths=output_lengths
            )
            epoch_val_losses.append(val_loss.item())

            # === DEBUG gate values for validation ===
            gate_out = outputs[2]   # [B, T]
            gate_sigmoid = torch.sigmoid(gate_out).detach().cpu()
            gate_stats_val.append([
                gate_sigmoid.mean().item(),
                gate_sigmoid.min().item(),
                gate_sigmoid.max().item()
            ])
            if val_sample is None:
                val_sample = {
                    'mel_true': mel_targets[0].detach().cpu().numpy(),
                    'mel_pred': outputs[1][0].detach().cpu().numpy(),
                }
                # Plot gate for first validation sample
                first_gate = torch.sigmoid(outputs[2][0]).detach().cpu().numpy()  # [T]
                first_gate_target = gate_targets[0].detach().cpu().numpy()  # [T]
                gate_path = os.path.join(VIS_DIR, f"gate_epoch{epoch}.png")
                plot_gate_outputs(
                    first_gate,
                    first_gate_target,
                    threshold=hparams["gate_threshold"],
                    out_path=gate_path
                )
                mlflow.log_artifact(gate_path, artifact_path="plots")
                # Print position of stop-frame (target==1) and gate value there
                stop_idx = (gate_targets[0] == 1).nonzero(as_tuple=True)[0]
                if len(stop_idx) > 0:
                    print(f"[Valid][Epoch {epoch}] Stop frame idx: {stop_idx[0].item()} | Gate sigmoid at stop: {first_gate[stop_idx[0].item()]:.4f}")

    gate_stats_val = np.array(gate_stats_val)
    print(f"[Valid][Epoch {epoch}] Gate sigmoid mean: {gate_stats_val[:,0].mean():.4f} | min: {gate_stats_val[:,1].min():.4f} | max: {gate_stats_val[:,2].max():.4f}")

    avg_val_loss = np.mean(epoch_val_losses)
    val_loss_history.append(avg_val_loss)

    loss_curve_path = os.path.join(VIS_DIR, f"loss_curves_epoch{epoch}.png")
    plot_loss_curves(
        train_loss_history,
        val_loss_history,
        loss_curve_path
    )
    mlflow.log_artifact(loss_curve_path, artifact_path="plots")

    if epoch % hparams["checkpoint_interval"] == 0:
        torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, f"tacotron2_epoch{epoch}.pth"))
        print(f"Saved checkpoint at epoch {epoch}")

        if val_sample is not None:
            mel_compare_path = os.path.join(VIS_DIR, f"mel_compare_epoch{epoch}.png")
            plot_mel_spectrograms(
                val_sample['mel_true'],
                val_sample['mel_pred'],
                mel_compare_path
            )
            mlflow.log_artifact(mel_compare_path, artifact_path="plots")

    mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
    mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f}")

    # === EARLY STOPPING ===
    early_stopper(avg_val_loss, model)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        best_model_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
        mlflow.log_artifact(best_model_path, artifact_path="models")
        break

print("Training complete. Checkpoints saved in checkpoints/")
mlflow.pytorch.log_model(model, artifact_path="final_model")
