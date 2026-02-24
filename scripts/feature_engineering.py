import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, unix_timestamp, when
from pyspark.ml.feature import StandardScaler, OneHotEncoder, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline

def process_features():
    """
    Performs feature engineering on the cleaned Taxi dataset using PySpark Pipeline.
    """
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "processed", "yellow_taxi_trip_2023.parquet")
    output_file = os.path.join(base_dir, "data", "processed", "yellow_taxi_trip_2023_features.parquet")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
        
    print("Starting Feature Engineering with PySpark...")
    
    spark = SparkSession.builder \
        .appName("YellowTaxiFeatureEngineering") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    
    try:
        df = spark.read.parquet(input_file)
        
        # 1. Feature Extraction (Time)
        df = df.withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))
        df = df.withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime"))) # 1=Sunday, 7=Saturday in PySpark
        df = df.withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))
        
        # Duration in minutes
        duration_sec = unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")
        df = df.withColumn("trip_duration_min", duration_sec / 60.0)
        
        # Define Columns
        scale_cols = [
            'trip_distance', 
            'fare_amount', 
            'extra', 
            'mta_tax', 
            'tip_amount', 
            'tolls_amount', 
            'improvement_surcharge', 
            'congestion_surcharge', 
            'airport_fee',
            'passenger_count'
        ]
        
        cat_cols = [
            'VendorID', 
            'RatecodeID', 
            'payment_type', 
            'store_and_fwd_flag'
        ]
        
        # MLlib ML Pipeline:
        stages = []
        
        # 2. Categorical Encoding
        # StringIndexer -> OneHotEncoder
        encoded_cat_cols = []
        for c in cat_cols:
            indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            encoder = OneHotEncoder(inputCols=[f"{c}_idx"], outputCols=[f"{c}_ohe"], dropLast=False)
            stages += [indexer, encoder]
            encoded_cat_cols.append(f"{c}_ohe")
            
        # 3. Scaling (Vectorize first, then Scale)
        assembler = VectorAssembler(inputCols=scale_cols, outputCol="numerical_features", handleInvalid="skip")
        scaler = StandardScaler(inputCol="numerical_features", outputCol="scaled_features", withStd=True, withMean=True)
        stages += [assembler, scaler]
        
        # Final Assembler (if needed to combine all features into one vector, but we can also just save the columns)
        # We will keep the vector columns for downstream MLlib, but if we want flat columns, we can extract them.
        # usually saving the VectorUDT natively in Parquet is fine for PySpark.
        
        pipeline = Pipeline(stages=stages)
        print("Fitting Pipeline...")
        model = pipeline.fit(df)
        
        print("Transforming Data...")
        df_transformed = model.transform(df)
        
        # Drop intermediate index columns to save space
        cols_to_drop = [f"{c}_idx" for c in cat_cols] + ["numerical_features"]
        df_final = df_transformed.drop(*cols_to_drop)
        
        # 4. Save
        print(f"Writing features to Parquet: {output_file}")
        df_final.write.mode("overwrite").parquet(output_file)
        
        print("\n" + "="*50)
        print("COMPLETED")
        print("="*50)
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"\nAn error occurred during feature engineering: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    process_features()
