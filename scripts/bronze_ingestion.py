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
        print(f"Reading raw CSV from {input_file} explicitly via RDDs for low-level parallelization...")
        
        # 1. Read as RDD of text lines
        raw_rdd = spark.sparkContext.textFile(input_file)
        
        # 2. Extract header
        header = raw_rdd.first()
        
        # 3. Filter out the header and malformed rows using RDD transformations (parallel processing)
        # Specifically dropping rows that don't have the correct number of comma-separated columns
        expected_cols = len(header.split(","))
        
        # RDD Transformation: filter -> map
        data_rdd = raw_rdd.filter(lambda line: line != header) \
                          .map(lambda line: line.split(",")) \
                          .filter(lambda cols: len(cols) == expected_cols)
                          
        # 4. Convert back to DataFrame using the schema from the header
        from pyspark.sql.types import StructType, StructField, StringType
        
        schema_fields = [StructField(col_name.strip('\"'), StringType(), True) for col_name in header.split(",")]
        schema = StructType(schema_fields)
        
        df = spark.createDataFrame(data_rdd, schema=schema)
            
        count = df.count()
        print(f"Ingested {count} rows via RDD parallel processing.")
        
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
