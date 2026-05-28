"""
PySpark Delta Lake Medallion Pipeline
Bronze → Silver → Gold

Bronze : raw CSV ingestion
Silver : cleaned + NER entity extraction
Gold   : risk-scored, care-pathway-enriched records

Architected for Azure Databricks — runs locally with PySpark 3.5 + Delta Lake 3.2.
"""

import logging
import os
import sys
from pathlib import Path

# --- Windows fix: embed env vars before SparkSession import ---
if sys.platform == "win32":
    java_home = os.environ.get("JAVA_HOME", "C:/Program Files/Java/jdk-11")
    os.environ["JAVA_HOME"] = java_home
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DELTA_BASE = BASE_DIR / "data" / "delta"
RAW_CSV = BASE_DIR / "data" / "sample" / "mtsamples.csv"


def get_spark() -> SparkSession:
    """Build a local SparkSession with Delta Lake support."""
    builder = (
        SparkSession.builder.appName("ClinicalTriagePipeline")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# --------------------------------------------------------------------------- #
#  BRONZE — raw ingestion                                                       #
# --------------------------------------------------------------------------- #
def ingest_bronze(spark: SparkSession) -> None:
    """Read raw MTSamples CSV and write to Delta Bronze table."""
    logger.info("=== BRONZE: Ingesting raw data ===")

    bronze_path = str(DELTA_BASE / "bronze" / "clinical_notes")

    schema = StructType(
        [
            StructField("description", StringType(), True),
            StructField("medical_specialty", StringType(), True),
            StructField("sample_name", StringType(), True),
            StructField("transcription", StringType(), True),
            StructField("keywords", StringType(), True),
        ]
    )

    df = (
        spark.read.option("header", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .schema(schema)
        .csv(str(RAW_CSV))
    )

    df = (
        df.withColumn("note_id", F.expr("uuid()"))
        .withColumn("ingested_at", F.current_timestamp())
        .filter(F.col("transcription").isNotNull())
        .filter(F.length(F.col("transcription")) > 50)
    )

    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(bronze_path)
    )

    count = spark.read.format("delta").load(bronze_path).count()
    logger.info(f"Bronze: {count} records written to {bronze_path}")


# --------------------------------------------------------------------------- #
#  SILVER — clean + deduplicate                                                 #
# --------------------------------------------------------------------------- #
def transform_silver(spark: SparkSession) -> None:
    """Clean Bronze records and write to Silver Delta table."""
    logger.info("=== SILVER: Cleaning and transforming ===")

    bronze_path = str(DELTA_BASE / "bronze" / "clinical_notes")
    silver_path = str(DELTA_BASE / "silver" / "clinical_notes_clean")

    df = spark.read.format("delta").load(bronze_path)

    df_clean = (
        df
        # Normalise specialty
        .withColumn("specialty_norm", F.trim(F.upper(F.col("medical_specialty"))))
        # Strip extra whitespace from transcription
        .withColumn(
            "transcription_clean", F.regexp_replace(F.col("transcription"), r"\s+", " ")
        )
        # Truncate to 4000 chars for LLM window
        .withColumn(
            "transcription_truncated", F.expr("substring(transcription_clean, 1, 4000)")
        )
        # Text length feature
        .withColumn("note_length", F.length(F.col("transcription_clean")))
        # Drop duplicates on trimmed text
        .dropDuplicates(["transcription_truncated"]).withColumn(
            "processed_at", F.current_timestamp()
        )
    )

    (
        df_clean.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(silver_path)
    )

    count = spark.read.format("delta").load(silver_path).count()
    logger.info(f"Silver: {count} clean records written to {silver_path}")


# --------------------------------------------------------------------------- #
#  GOLD — entity + risk aggregation                                             #
# --------------------------------------------------------------------------- #
def build_gold(spark: SparkSession) -> None:
    """
    Simulate Gold-layer aggregation.
    In production this calls the NER + LangGraph pipeline per partition.
    Here we write a placeholder Gold schema ready for API consumption.
    """
    logger.info("=== GOLD: Building aggregated risk table ===")

    silver_path = str(DELTA_BASE / "silver" / "clinical_notes_clean")
    gold_path = str(DELTA_BASE / "gold" / "triage_results")

    df = spark.read.format("delta").load(silver_path)

    # Placeholder risk score — replaced by actual LangGraph output in production
    df_gold = (
        df.withColumn("risk_score", F.lit(None).cast(FloatType()))
        .withColumn("risk_level", F.lit("PENDING"))
        .withColumn("care_pathway", F.lit("PENDING"))
        .withColumn("entities_json", F.lit("{}"))
        .withColumn("scored_at", F.current_timestamp())
        .select(
            "note_id",
            "specialty_norm",
            "transcription_truncated",
            "note_length",
            "risk_score",
            "risk_level",
            "care_pathway",
            "entities_json",
            "scored_at",
        )
    )

    (
        df_gold.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(gold_path)
    )

    count = spark.read.format("delta").load(gold_path).count()
    logger.info(f"Gold: {count} records written to {gold_path}")


# --------------------------------------------------------------------------- #
#  MAIN                                                                         #
# --------------------------------------------------------------------------- #
def run_pipeline() -> None:
    spark = get_spark()
    try:
        ingest_bronze(spark)
        transform_silver(spark)
        build_gold(spark)
        logger.info("=== Pipeline complete: Bronze → Silver → Gold ===")
    finally:
        spark.stop()


if __name__ == "__main__":
    run_pipeline()
