from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import os
import sys

def train_models():
    """
    Trains and evaluates models using PySpark.
    """
    
    # Initialize Spark
    # memory configuration might be needed for RF on large data
    spark = SparkSession.builder \
        .appName("TaxiFarePrediction") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "train.parquet")
    val_path = os.path.join(base_dir, "data", "processed", "validation.parquet")
    
    if not os.path.exists(train_path):
        print(f"Error: Train file {train_path} not found.")
        sys.exit(1)
        
    print("Loading data...")
    train_data = spark.read.parquet(train_path)
    val_data = spark.read.parquet(val_path)
    
    # Define Feature Columns
    # We explicitly select features to avoid leakage (e.g. not including total_amount or tips)
    # And we exclude the target 'fare_amount'
    
    # Get all columns
    all_cols = train_data.columns
    
    # Columns to Exclude
    exclude_cols = {
        'tpep_pickup_datetime', 'tpep_dropoff_datetime', # Dates
        'fare_amount', # Target
        'total_amount', # Leakage (Target + Tips)
        'tip_amount', 'tolls_amount', 'mta_tax', 'extra', # Post-trip costs / Leakage
        'improvement_surcharge', 'congestion_surcharge', 'airport_fee',
        'PULocationID', 'DOLocationID' # Exclude High Card IDs for this simple run
    }
    
    # Feature List
    feature_cols = [c for c in all_cols if c not in exclude_cols]
    
    print(f"\nTarget: fare_amount")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # Vector Assembler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    # Transform Data
    # Note: handleInvalid='skip' is safer if any nulls slipped through, though we cleaned them.
    print("Preparing feature vectors...")
    train_data = assembler.setHandleInvalid("skip").transform(train_data)
    val_data = assembler.setHandleInvalid("skip").transform(val_data)
    
    # Optimization: Cache data in memory
    train_data = train_data.select("features", "fare_amount").cache()
    val_data = val_data.select("features", "fare_amount").cache()
    
    # Evaluator
    evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")
    
    print("\n" + "="*50)
    print("MODEL 1: LINEAR REGRESSION")
    print("="*50)
    lr = LinearRegression(labelCol="fare_amount", featuresCol="features")
    lr_model = lr.fit(train_data)
    
    lr_predictions = lr_model.transform(val_data)
    lr_rmse = evaluator.evaluate(lr_predictions)
    print(f"Linear Regression RMSE: {lr_rmse:.4f}")
    
    print("\n" + "="*50)
    print("MODEL 2: RANDOM FOREST REGRESSOR")
    print("="*50)
    # Limiting depth and trees for speed on this large dataset demo
    rf = RandomForestRegressor(labelCol="fare_amount", featuresCol="features", numTrees=20, maxDepth=10)
    rf_model = rf.fit(train_data)
    
    rf_predictions = rf_model.transform(val_data)
    rf_rmse = evaluator.evaluate(rf_predictions)
    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    
    # Save Best Model (Example: RF)
    model_path = os.path.join(base_dir, "results", "rf_model")
    rf_model.write().overwrite().save(model_path)
    print(f"\nBest model saved to {model_path}")
    
    spark.stop()

if __name__ == "__main__":
    train_models()
