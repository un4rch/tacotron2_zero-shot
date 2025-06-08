# tacotron2_zero-shot

```bash
source install.sh
```

## Entrenamiento
```bash
python train.py
```

## Inferencia (servidor FastAPI)

```bash
uvicorn infer_api:app --reload
```

```bash
curl -X POST "http://localhost:8000/synthesize" \
  -F "text=Hello, I am Unai" \
  -F "speaker_wavs=Voz1.wav" \
  -F "speaker_wavs=Voz2.wav" \
  -F "speaker_wavs=Voz3.wav" \
  --output resultado.wav
```
