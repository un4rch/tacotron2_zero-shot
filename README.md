# tacotron2_zero-shot

```bash
source install.sh
```

## Spark y cassandra

```sql
CREATE KEYSPACE IF NOT EXISTS testks
WITH replication = {
  'class': 'SimpleStrategy',
  'replication_factor': 1
};
USE testks;
CREATE TABLE IF NOT EXISTS metadata (
  wav_path TEXT PRIMARY KEY,
  transcript TEXT,
  speaker_id TEXT
);
```

```bash
$SPARK_HOME/bin/spark-submit \
  --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \
  spark_cassandra.py
```

## Entrenamiento
```bash
#python train.py
python -m torch.distributed.run --nproc_per_node=2 train.py
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
