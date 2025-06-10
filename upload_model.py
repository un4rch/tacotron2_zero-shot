import torch
import mlflow
import mlflow.pytorch
import os
import sys
import json
from collections import OrderedDict

# 0. Lee hiperparámetros
with open("hparams.json", "r") as f:
    hparams = json.load(f)

# 1. Configura MLflow
MLFLOW_TRACKING_URI = "http://admin:mlflow_password@mlflow.172.16.57.20.nip.io/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Tacotron2_Zero-Shot")

# 2. Fuerza CPU
device = torch.device("cpu")

# 3. Importa y crea el modelo sobre CPU
BASE_DIR = os.path.join(os.path.dirname(__file__),
    "DeepLearningExamples", "PyTorch", "SpeechSynthesis", "Tacotron2", "tacotron2")
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
).to(device)

# 4. Carga el checkpoint en CPU
checkpoint = torch.load("./checkpoints/best_model.pth", map_location="cpu")
raw_state_dict = checkpoint.get("model_state_dict", checkpoint)

# 4.1. Strip “module.” prefix
new_state_dict = OrderedDict()
for k, v in raw_state_dict.items():
    name = k.replace("module.", "")  # quita el “module.” inicial
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

# 5. Loggea en MLflow
with mlflow.start_run(run_name="run_2171956") as run:
    mlflow.log_params(hparams)
    mlflow.log_param("device", str(device))
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("weight_decay", 1e-5)
    mlflow.log_param("guided_attn_loss_weight", 0.2)

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="final_model"
    )

print("Modelo registrado en:", run.info.artifact_uri + "/final_model")
