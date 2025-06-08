import os
import torch
import torchaudio
import numpy as np
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from tempfile import NamedTemporaryFile
from speechbrain.inference.speaker import EncoderClassifier
import sys

# ==== Ajusta estas rutas según tus necesidades ====
HPARAMS_PATH = "hparams.json"
DEFAULT_CHECKPOINT = "checkpoints/tacotron2_epoch9.pth"
SPEAKER_WAVS_DIR = "/home/uelorriaga/tts_lora/data/my_voice/wavs/"

# ==== Carga hparams y utilidades ====
with open(HPARAMS_PATH, "r") as f:
    hparams = json.load(f)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
tts_utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

# ==== FastAPI App ====
app = FastAPI()

def prepare_text(text, tts_utils):
    text = text.lower()
    sequences, input_lengths = tts_utils.prepare_input_sequence([text])
    if torch.is_tensor(sequences):
        sequences = sequences.cpu().numpy()
    if torch.is_tensor(input_lengths):
        input_lengths = input_lengths.cpu().numpy()
    text_padded = torch.LongTensor(sequences)
    input_lengths = torch.LongTensor(input_lengths)
    return text_padded, input_lengths

def get_speaker_embedding(wav_paths, classifier):
    embs = []
    for wav_path in wav_paths:
        waveform, sr = torchaudio.load(wav_path)
        emb = classifier.encode_batch(waveform.to(DEVICE)).detach().cpu().squeeze(0).mean(dim=0)
        embs.append(emb)
    emb = torch.stack(embs, dim=0).mean(dim=0)
    return emb.to(DEVICE)

def infer_tts(text, speaker_wavs, tacotron2_ckpt, output_path):
    text_padded, input_lengths = prepare_text(text, tts_utils)
    text_padded = text_padded.to(DEVICE)
    input_lengths = input_lengths.to(DEVICE)

    spk_embedding = get_speaker_embedding(speaker_wavs, classifier)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "DeepLearningExamples", "PyTorch", "SpeechSynthesis", "Tacotron2", "tacotron2"))
    sys.path.append(BASE_DIR)
    from model import Tacotron2
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
        audio = waveglow.infer(mel, sigma=1.0)
        audio = audio.cpu().numpy()[0]
        audio = audio / np.max(np.abs(audio))
        torchaudio.save(output_path, torch.from_numpy(audio).unsqueeze(0), hparams.get("sample_rate", 22050))

@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    speaker_wavs: list[str] = Form(...),   # ahora espera nombres de archivo, NO archivos
    checkpoint: UploadFile = File(None)
):
    # Construir las rutas reales de los wav
    wav_paths = [os.path.join(SPEAKER_WAVS_DIR, fname) for fname in speaker_wavs]

    # Chequear que existan los archivos y dar error útil si falta alguno
    for path in wav_paths:
        if not os.path.exists(path):
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=f"Archivo de voz no encontrado: {path}")

    # Checkpoint temporal (igual que antes)
    if checkpoint is not None:
        with NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
            tmp.write(await checkpoint.read())
            checkpoint_path = tmp.name
    else:
        checkpoint_path = DEFAULT_CHECKPOINT

    # Archivo de salida temporal
    with NamedTemporaryFile(delete=False, suffix=".wav") as out_tmp:
        output_path = out_tmp.name

    # Ejecutar inferencia
    infer_tts(text, wav_paths, checkpoint_path, output_path)

    if checkpoint is not None:
        os.remove(checkpoint_path)

    return FileResponse(output_path, filename="output.wav", media_type="audio/wav")
