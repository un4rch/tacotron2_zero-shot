#!/bin/bash

TACO_DIR="DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/tacotron2"
if [ ! -d "$TACO_DIR" ]; then
    echo "Cloning Tacotron2 repository..."
    git clone https://github.com/NVIDIA/DeepLearningExamples.git
fi
echo "Copying model.py to $TACO_DIR ..."
mkdir -p "$TACO_DIR"
cp -f model.py "$TACO_DIR"/

FILE="train-clean-100.tar.gz"
DIR="data/LibriTTS/"

# Check if the file exists
if [ ! -f "$FILE" ]; then
    echo "$FILE not found. Downloading..."
    wget https://www.openslr.org/resources/60/$FILE
else
    echo "$FILE already exists. Skipping download."
fi

# Extract if not already extracted
if [ ! -d "${DIR}LibriTTS/train-clean-100" ]; then
    mkdir -p "$DIR"
    tar -xzvf "$FILE" -C "$DIR"
else
    echo "Data already extracted in $DIR. Skipping extraction."
fi

#rm "$FILE"

wget https://downloads.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
tar xzf spark-3.5.1-bin-hadoop3.tgz -C $HOME
export SPARK_HOME="$HOME/spark-3.5.1-bin-hadoop3"
export PATH="$SPARK_HOME/bin:$PATH"

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python make_metadata.py

#python train.py

#python infer.py --checkpoint checkpoints/tacotron2_epoch9.pth --text "Hello, I am Unai" --speaker_wav data/my_voice/wavs/Voz1.wav data/my_voice/wavs/Voz2.wav data/my_voice/wavs/Voz3.wav --out_wav ex.wav
