from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, input_file_name, regexp_extract
from pyspark.sql.types import StringType
import os

# CONFIGURACIÓN
LIBRITTS_ROOT = "data/LibriTTS/LibriTTS/train-clean-100"

# 1. Crear SparkSession con Cassandra Connector
spark = SparkSession.builder \
    .appName("LibriTTS_Metadata_Extractor") \
    .master("local[*]") \
    .config("spark.cassandra.connection.host", "127.0.0.1") \
    .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0") \
    .getOrCreate()

# 2. Cargar archivos .normalized.txt (que contienen transcripciones)
transcripts_df = spark.read.text(f"{LIBRITTS_ROOT}/*/*/*.normalized.txt") \
    .withColumn("transcript", input_file_name()) \
    .withColumnRenamed("value", "text")

# 3. UDF para convertir path del .txt a path del .wav
@udf(returnType=StringType())
def normalized_to_wav_path(txt_path):
    wav_path = txt_path.replace(".normalized.txt", ".wav")
    return wav_path

# 4. UDF para extraer speaker ID
@udf(returnType=StringType())
def extract_speaker(path):
    return path.split(os.sep)[-3]

# 5. Generar columnas necesarias
processed_df = transcripts_df \
    .withColumn("wav_path", normalized_to_wav_path("transcript")) \
    .withColumn("speaker_id", extract_speaker("transcript")) \
    .select("wav_path", "text", "speaker_id") \
    .withColumnRenamed("text", "transcript")

# OPCIONAL: mostrar ejemplo
processed_df.show(5, truncate=False)

# 6. Guardar en Cassandra (revisar tabla antes)
processed_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="metadata", keyspace="testks") \
    .save()

spark.stop()
