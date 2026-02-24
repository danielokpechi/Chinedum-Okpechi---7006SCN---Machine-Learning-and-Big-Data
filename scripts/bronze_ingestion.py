import sys
import os
from pyspark.sql import SparkSession

def ingest_bronze():
    """
    Bronze Layer Ingestion:
    - Reads raw CSV using Spark (treating as RDD/Text first to handle malformed rows if needed, 
      or using permissive CSV reader).
    - Saves as Parquet to data/bronze.
    - Minimal validation (dropped malformed lines).
    """
    print("\n[BRONZE] Starting Ingestion...")

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "raw", "yellow_taxi_trip_2023.csv")
    output_dir = os.path.join(base_dir, "data", "bronze", "yellow_tripdata_2023.parquet")

    if not os.path.exists(input_file):
        print(f"Error: Raw input file not found at {input_file}")
        sys.exit(1)

    spark = SparkSession.builder \
        .appName("Bronze_Ingestion") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    try:
        print(f"Reading raw CSV from {input_file}...")
        
        # Read with schema inference disabled (read as strings) just like a true raw landing zone.
        # mode="DROPMALFORMED" ensures we handle any parsing errors by skipping bad lines.
        df = spark.read.option("header", "true") \
            .option("inferSchema", "false") \
            .option("mode", "DROPMALFORMED") \
            .csv(input_file)
            
        count = df.count()
        print(f"Ingested {count} rows.")
        
        print(f"Saving to Bronze (Parquet): {output_dir}")
        df.write.mode("overwrite").parquet(output_dir)
        print("[BRONZE] Ingestion Complete.")
        
    except Exception as e:
        print(f"[BRONZE] Error: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    ingest_bronze()
