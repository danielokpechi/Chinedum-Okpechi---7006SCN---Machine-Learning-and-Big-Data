import sys
import os
from pyspark.sql import SparkSession

def ingest_zones():
    """
    Ingest Taxi Zones (Reference Data) to Bronze.
    - Reads data/raw/taxi_zone_lookup.csv
    - Writes to data/bronze/taxi_zones.parquet
    """
    print("\n[BRONZE] Ingesting Taxi Zones...")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "raw", "taxi_zone_lookup.csv")
    output_dir = os.path.join(base_dir, "data", "bronze", "taxi_zones.parquet")

    if not os.path.exists(input_file):
        print(f"Error: Zones file not found at {input_file}")
        sys.exit(1)

    spark = SparkSession.builder \
        .appName("Zone_Ingestion") \
        .getOrCreate()

    try:
        df = spark.read.option("header", "true").csv(input_file)
        print(f"Ingested {df.count()} zones.")
        
        print(f"Saving to Bronze: {output_dir}")
        df.write.mode("overwrite").parquet(output_dir)
        print("[BRONZE] Zones Ingestion Complete.")
        
    except Exception as e:
        print(f"[BRONZE] Error: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    ingest_zones()
