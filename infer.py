import os
import sys
import torch
import torchaudio
import numpy as np
import json

from speechbrain.inference.speaker import EncoderClassifier

# ====== Path Setup ======
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "DeepLearningExamples", "PyTorch", "SpeechSynthesis", "Tacotron2", "tacotron2"))
sys.path.append(BASE_DIR)
from model import Tacotron2

# ====== Hparams Path (puedes dejarlo igual) ======
HPARAMS_PATH = "hparams.json"

with open(HPARAMS_PATH, "r") as f:
    hparams = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== TTS utils ======
tts_utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils")

# ====== Speaker Encoder ======
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

# ====== MelSpectrogram transform ======
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=hparams.get("sample_rate", 22050),
    n_fft=hparams.get("n_fft", 1024),
    win_length=hparams.get("win_length", 1024),
    hop_length=hparams.get("hop_length", 256),
    n_mels=hparams["n_mel_channels"],
    f_min=0.0,
    f_max=hparams.get("f_max", 8000)
).to(DEVICE)

def prepare_text(text, tts_utils):
    text = text.lower()
    sequences, input_lengths = tts_utils.prepare_input_sequence([text])
    if torch.is_tensor(sequences):
        sequences = sequences.cpu().numpy()
    if torch.is_tensor(input_lengths):
        input_lengths = input_lengths.cpu().numpy()
    text_padded = torch.LongTensor(sequences)     # [1, T]
    input_lengths = torch.LongTensor(input_lengths)   # [1]
    return text_padded, input_lengths

def get_speaker_embedding(wav_paths, classifier):
    if isinstance(wav_paths, str):
        wav_paths = [wav_paths]
    embs = []
    for wav_path in wav_paths:
        waveform, sr = torchaudio.load(wav_path)
        emb = classifier.encode_batch(waveform.to(DEVICE)).detach().cpu().squeeze(0).mean(dim=0)
        embs.append(emb)
    emb = torch.stack(embs, dim=0).mean(dim=0)  # media si varios
    return emb.to(DEVICE)                       # [192]

def infer_tts(text, speaker_wavs, out_wav, tacotron2_ckpt):
    # === INPUTS ===
    text_padded, input_lengths = prepare_text(text, tts_utils)
    text_padded = text_padded.to(DEVICE)
    input_lengths = input_lengths.to(DEVICE)

    # === Speaker embedding ===
    spk_embedding = get_speaker_embedding(speaker_wavs, classifier) # [1, 192]
    print("Speaker embedding shape (should be [1, 192]):", spk_embedding.shape)

    # === Load Tacotron2 Model ===
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
    checkpoint = torch.load(tacotron2_ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    # === Vocoder ===
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to(DEVICE)
    waveglow.eval()
    for m in waveglow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')
    for p in waveglow.parameters():
        p.requires_grad = False

    with torch.no_grad():
        mel, _, _ = model.infer(text_padded, input_lengths, speaker_embedding=spk_embedding)
        print("Mel shape:", mel.shape)
        audio = waveglow.infer(mel, sigma=1.0)
        audio = audio.cpu().numpy()[0]
        audio = audio / np.max(np.abs(audio))
        torchaudio.save(out_wav, torch.from_numpy(audio).unsqueeze(0), hparams.get("sample_rate", 22050))
        print(f"Guardado {out_wav}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--speaker_wav', type=str, nargs='+', required=True)
    parser.add_argument('--out_wav', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    infer_tts(args.text, args.speaker_wav, args.out_wav, args.checkpoint)
