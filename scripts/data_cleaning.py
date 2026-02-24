import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, dropna
from pyspark.sql.types import IntegerType, FloatType, StringType

def clean_data():
    """
    Reads the raw Yellow Taxi Trip CSV, cleans it using PySpark, and saves to Parquet.
    Cleaning steps:
    1. Parse dates.
    2. Drop rows with missing values in key columns.
    3. Drop duplicates.
    4. Save to Parquet.
    """
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "raw", "yellow_taxi_trip_2023.csv")
    output_file = os.path.join(base_dir, "data", "processed", "yellow_taxi_trip_2023.parquet")
    
    # Check inputs
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
        
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Starting data cleaning process with PySpark...")
    print(f"Input: {input_file}")
    
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("YellowTaxiDataCleaning") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
        
    try:
        # 1. Read CSV using PySpark
        df = spark.read.csv(input_file, header=True, inferSchema=True)
        
        initial_count = df.count()
        print(f"Total Rows Read: {initial_count}")
        
        # 2. Convert Timestamps
        # Format: 01/01/2023 12:32:10 AM -> MM/dd/yyyy hh:mm:ss a
        df = df.withColumn("tpep_pickup_datetime", to_timestamp(col("tpep_pickup_datetime"), "MM/dd/yyyy hh:mm:ss a"))
        df = df.withColumn("tpep_dropoff_datetime", to_timestamp(col("tpep_dropoff_datetime"), "MM/dd/yyyy hh:mm:ss a"))
        
        # 3. Cast columns to proper types to match schema
        df = df \
            .withColumn("passenger_count", col("passenger_count").cast(IntegerType())) \
            .withColumn("RatecodeID", col("RatecodeID").cast(IntegerType())) \
            .withColumn("VendorID", col("VendorID").cast(IntegerType())) \
            .withColumn("payment_type", col("payment_type").cast(IntegerType())) \
            .withColumn("PULocationID", col("PULocationID").cast(IntegerType())) \
            .withColumn("DOLocationID", col("DOLocationID").cast(IntegerType())) \
            .withColumn("store_and_fwd_flag", col("store_and_fwd_flag").cast(StringType())) \
            .withColumn("fare_amount", col("fare_amount").cast(FloatType())) \
            .withColumn("extra", col("extra").cast(FloatType())) \
            .withColumn("mta_tax", col("mta_tax").cast(FloatType())) \
            .withColumn("tip_amount", col("tip_amount").cast(FloatType())) \
            .withColumn("tolls_amount", col("tolls_amount").cast(FloatType())) \
            .withColumn("improvement_surcharge", col("improvement_surcharge").cast(FloatType())) \
            .withColumn("total_amount", col("total_amount").cast(FloatType())) \
            .withColumn("congestion_surcharge", col("congestion_surcharge").cast(FloatType())) \
            .withColumn("airport_fee", col("airport_fee").cast(FloatType())) \
            .withColumn("trip_distance", col("trip_distance").cast(FloatType()))
            
        # 4. Drop Missing Values in Subset
        subset_cols = [
            'tpep_pickup_datetime', 
            'tpep_dropoff_datetime', 
            'passenger_count', 
            'RatecodeID',
            'store_and_fwd_flag'
        ]
        df_cleaned = df.na.drop(subset=subset_cols)
        
        dna_count = df_cleaned.count()
        print(f"Dropped (NaN): {initial_count - dna_count}")
        
        # 5. Drop Duplicates
        df_cleaned = df_cleaned.dropDuplicates()
        
        final_count = df_cleaned.count()
        print(f"Dropped (Dupes): {dna_count - final_count}")
        print(f"Total Rows to Save: {final_count}")
        
        # 6. Write to Parquet
        print(f"Writing to Parquet: {output_file}")
        df_cleaned.write.mode("overwrite").parquet(output_file)
        
        print("\n" + "="*50)
        print("COMPLETED")
        print("="*50)
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"\nAn error occurred during cleaning: {e}")
        sys.exit(1)
        
    finally:
        spark.stop()

if __name__ == "__main__":
    clean_data()
