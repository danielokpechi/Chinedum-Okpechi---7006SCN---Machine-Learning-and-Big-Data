import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, dayofweek, unix_timestamp, when, lit, broadcast
from pyspark.sql.types import DoubleType

def process_silver():
    """
    Silver Layer Processing:
    - Reads Bronze Parquet.
    - Cleaning: Types, Nulls, Duplicates.
    - Feature Engineering: Time features, Scaling, OHE.
    - Saves to data/silver.
    """
    print("\n[SILVER] Starting Processing...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "bronze", "yellow_tripdata_2023.parquet")
    zones_path = os.path.join(base_dir, "data", "bronze", "taxi_zones.parquet") # [NEW]
    output_path = os.path.join(base_dir, "data", "silver", "taxi_features.parquet")
    
    spark = SparkSession.builder \
        .appName("Silver_Processing") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
        
    try:
        # 1. READ BRONZE
        print("Reading Bronze data...")
        df = spark.read.parquet(input_path)
        
        # TEST MODE: Limit rows for quick verification
        if os.environ.get("IS_TEST_RUN"):
            print("⚠️ TEST MODE DETECTED: Limiting to 100,000 rows.")
            df = df.limit(100000)
            
        # 1.1 READ ZONES (Dimension Table)
        if os.path.exists(zones_path):
            print("Reading Zone data for Broadcast Join...")
            zones_df = spark.read.parquet(zones_path)
            # Join Logic (PULocationID -> LocationID)
            # We use Broadcast Join as per assignment requirement for small tables
            print("Performing Broadcast Join with Zones...")
            df = df.join(broadcast(zones_df), df.PULocationID == zones_df.LocationID, "left") \
                   .drop("LocationID", "service_zone") \
                   .withColumnRenamed("Borough", "pickup_borough") \
                   .withColumnRenamed("Zone", "pickup_zone")
        else:
            print("⚠️ Warning: Zone data not found. Skipping Broadcast Join.")
        
        # 2. CLEANING
        print("Cleaning data (Casting, Nulls, Dupes)...")
        
        # Casting typical columns
        # Note: Input format is %m/%d/%Y %I:%M:%S %p based on previous script analysis
        df_casted = df \
            .withColumn("tpep_pickup_datetime", to_timestamp(col("tpep_pickup_datetime"), "MM/dd/yyyy hh:mm:ss a")) \
            .withColumn("tpep_dropoff_datetime", to_timestamp(col("tpep_dropoff_datetime"), "MM/dd/yyyy hh:mm:ss a")) \
            .withColumn("passenger_count", col("passenger_count").cast(DoubleType())) \
            .withColumn("trip_distance", col("trip_distance").cast(DoubleType())) \
            .withColumn("fare_amount", col("fare_amount").cast(DoubleType())) \
            .withColumn("total_amount", col("total_amount").cast(DoubleType())) \
            .withColumn("VendorID", col("VendorID").cast(DoubleType())) \
            .withColumn("RatecodeID", col("RatecodeID").cast(DoubleType())) \
            .withColumn("payment_type", col("payment_type").cast(DoubleType())) \
            .withColumn("extra", col("extra").cast(DoubleType())) \
            .withColumn("mta_tax", col("mta_tax").cast(DoubleType())) \
            .withColumn("tip_amount", col("tip_amount").cast(DoubleType())) \
            .withColumn("tolls_amount", col("tolls_amount").cast(DoubleType())) \
            .withColumn("improvement_surcharge", col("improvement_surcharge").cast(DoubleType())) \
            .withColumn("congestion_surcharge", col("congestion_surcharge").cast(DoubleType())) \
            .withColumn("airport_fee", col("airport_fee").cast(DoubleType()))

        # Drop duplicates
        df_dedup = df_casted.dropDuplicates()
        
        # Drop Nulls in critical columns
        critical_cols = [
            'tpep_pickup_datetime', 'tpep_dropoff_datetime', 
            'passenger_count', 'trip_distance', 'fare_amount'
        ]
        df_clean = df_dedup.dropna(subset=critical_cols)
        
        # 3. FEATURE ENGINEERING
        print("Feature Engineering (Time, OHE, Scaling)...")
        
        # Time Features
        df_feat = df_clean \
            .withColumn("pickup_hour", hour(col("tpep_pickup_datetime")).cast(DoubleType())) \
            .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")).cast(DoubleType())) \
            .withColumn("is_weekend", when(col("day_of_week") >= 6, 1.0).otherwise(0.0)) \
            .withColumn("trip_duration_min", 
                        (unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60.0)
                        
        # Filter logical inconsistencies
        df_feat = df_feat.filter(
            (col("trip_duration_min") > 0) & 
            (col("trip_distance") > 0) & 
            (col("fare_amount") > 0)
        )
        
        # SCALING
        # OPTIMIZATION: We do NOT scale here. We let the Gold Machine Learning Pipeline handle scaling.
        # This ensures the scaling parameters (mean/std) are saved with the model for correct inference.
        # We just keep the raw "business" units (miles, minutes, etc.) in Silver.

        
        # ONE HOT ENCODING
        # Manual OHE to match previous structured schema (e.g. VendorID_1, VendorID_2)
        # This is robust and ensures Gold layer finds exactly the columns it expects.
        
        # VendorID (1, 2)
        df_feat = df_feat \
            .withColumn("VendorID_1", when(col("VendorID") == 1, 1.0).otherwise(0.0)) \
            .withColumn("VendorID_2", when(col("VendorID") == 2, 1.0).otherwise(0.0))
            
        # Payment Type (1..4)
        for i in range(1, 6):
            df_feat = df_feat.withColumn(f"payment_type_{i}", when(col("payment_type") == i, 1.0).otherwise(0.0))

        # Store and Fwd (Y/N)
        # In Bronze, it was string. casted to double above might have failed if it was 'Y'/'N' string.
        # Wait, I casted everything to Double above? 
        # `withColumn("store_and_fwd_flag", col("store_and_fwd_flag").cast(DoubleType()))` was NOT in my cast list above.
        # Good, keep it as string.
        df_feat = df_feat \
            .withColumn("store_and_fwd_flag_Y", when(col("store_and_fwd_flag") == 'Y', 1.0).otherwise(0.0)) \
            .withColumn("store_and_fwd_flag_N", when(col("store_and_fwd_flag") == 'N', 1.0).otherwise(0.0))
            
        # RatecodeID (1..6 + 99)
        # Assuming common IDs
        for i in [1, 2, 3, 4, 5, 6, 99]:
            df_feat = df_feat.withColumn(f"RatecodeID_{i}", when(col("RatecodeID") == i, 1.0).otherwise(0.0))

        # Save
        print(f"Saving to Silver (Parquet): {output_path}")
        df_feat.write.mode("overwrite").parquet(output_path)
        print("[SILVER] Processing Complete.")
        
    except Exception as e:
        print(f"[SILVER] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    process_silver()
