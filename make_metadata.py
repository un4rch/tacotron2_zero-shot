import os
import glob
import csv

# data/LibriTTS/LibriTTS/train-clean-100/<speaker_id>/<chapter_id>/<file>.normalized.txt

# Cambia esta ruta según tu estructura si lo necesitas
root_dir = "data/LibriTTS/LibriTTS/train-clean-100"
metadata_path = "data/LibriTTS/metadata.csv"

wav_files = glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
entries = []

for wav_path in wav_files:
    parts = wav_path.split(os.sep)
    speaker_id = parts[-3]  # '103'
    transcript_dir = os.path.dirname(wav_path)
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    transcript_file = os.path.join(transcript_dir, f"{basename}.normalized.txt")
    transcript = ""
    if os.path.exists(transcript_file):
        with open(transcript_file, encoding="utf-8") as f:
            transcript = f.read().strip()
    if transcript:
        entries.append([os.path.abspath(wav_path), transcript, speaker_id])
        #entries.append([os.path.splitext(os.path.abspath(wav_path))[0], transcript, speaker_id])

print(f"Found {len(entries)} samples. Writing metadata...")
with open(metadata_path, "w", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter='|')
    for entry in entries:
        writer.writerow(entry)
print(f"metadata.csv written at {metadata_path}")
