
import os
import sys
import json
import math
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.regression import GBTRegressionModel, RandomForestRegressionModel, LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, LongType, IntegerType, StringType, StructType, StructField

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEST_PATH = os.path.join(DATA_DIR, "test.parquet")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "mllib_best_model")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "bootstrap_rmse_stats.json")

def create_spark_session():
    return SparkSession.builder \
        .appName("Bootstrap_RMSE_Analysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.memory", "4g") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
        .getOrCreate()

def get_data_schema(spark, path):
    # Workaround for [PARQUET_TYPE_ILLEGAL] INT64 (TIMESTAMP(NANOS,false))
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema.to_arrow_schema()
    
    spark_fields = []
    feature_candidates = []
    
    for field in schema:
        name = field.name
        dtype = str(field.type)
        
        if 'timestamp' in dtype or 'int64' in dtype: spark_type = LongType()
        elif 'int32' in dtype: spark_type = IntegerType()
        elif 'double' in dtype: spark_type = DoubleType()
        elif 'string' in dtype: spark_type = StringType()
        else: spark_type = DoubleType()
            
        spark_fields.append(StructField(name, spark_type, True))
        
        exclude_feature_cols = {
            'fare_amount', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 
            'improvement_surcharge', 'congestion_surcharge', 'airport_fee', 
            'total_amount', 'tip_amount', 'tolls_amount', 'mta_tax', 'extra', 
            'PULocationID', 'DOLocationID', '__index_level_0__'
        }
        
        if name not in exclude_feature_cols and (
            isinstance(spark_type, (IntegerType, LongType, DoubleType))
        ) and 'datetime' not in name:
            feature_candidates.append(name)
            
    return StructType(spark_fields), feature_candidates

def load_model():
    # Try different model types just in case, though we know it's likely GBT
    try:
        return GBTRegressionModel.load(BEST_MODEL_PATH)
    except:
        try:
            return RandomForestRegressionModel.load(BEST_MODEL_PATH)
        except:
            return LinearRegressionModel.load(BEST_MODEL_PATH)

def run_analysis():
    spark = create_spark_session()
    
    try:
        print(f"Loading Test Data from {TEST_PATH}...")
        if not os.path.exists(TEST_PATH):
            print("Test data not found!")
            return

        schema, feature_cols = get_data_schema(spark, TEST_PATH)
        df = spark.read.schema(schema).parquet(TEST_PATH)
        
        # Assemble Features
        print("Assembling features...")
        for c in feature_cols:
            df = df.withColumn(c, col(c).cast(DoubleType()))
            
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        test_data = assembler.transform(df).select("features", "fare_amount")
        
        # Load Model
        print(f"Loading Best Model from {BEST_MODEL_PATH}...")
        model = load_model()
        
        # Generate Predictions
        print("Generating predictions...")
        predictions = model.transform(test_data)
        
        # Convert to NumPy (Lite version)
        print("Collecting predictions to Python/NumPy (prediction, fare_amount only)...")
        # Ensure we only select necessary cols to save memory
        results = predictions.select("prediction", "fare_amount").collect()
        
        preds_array = np.array([row["prediction"] for row in results], dtype=np.float32)
        actuals_array = np.array([row["fare_amount"] for row in results], dtype=np.float32)
        
        n_rows = len(preds_array)
        print(f"Collected {n_rows} rows.")
        
        # Bootstrap
        print("Starting Bootstrap Analysis (100 iterations)...")
        n_bootstraps = 100
        rmse_values = []
        
        np.random.seed(42)
        
        for i in range(n_bootstraps):
            # Sample with replacement using NumPy
            indices = np.random.choice(n_rows, size=n_rows, replace=True)
            
            sample_preds = preds_array[indices]
            sample_actuals = actuals_array[indices]
            
            # Calculate RMSE
            # RMSE = sqrt(mean((pred - actual)^2))
            mse = np.mean((sample_preds - sample_actuals) ** 2)
            rmse = math.sqrt(mse)
            rmse_values.append(rmse)
            
            if (i+1) % 10 == 0:
                print(f"  Iteration {i+1}/{n_bootstraps}: RMSE={rmse:.4f}")
                
        # Stats
        rmse_mean = np.mean(rmse_values)
        rmse_std = np.std(rmse_values)
        ci_lower = np.percentile(rmse_values, 2.5)
        ci_upper = np.percentile(rmse_values, 97.5)
        
        print("\n" + "="*50)
        print("BOOTSTRAP RESULTS")
        print("="*50)
        print(f"Bootstrap RMSE Mean: {rmse_mean:.5f}")
        print(f"Bootstrap RMSE Std:  {rmse_std:.5f}")
        print(f"95% CI Lower:        {ci_lower:.5f}")
        print(f"95% CI Upper:        {ci_upper:.5f}")
        print("="*50)
        
        # Save Results
        results = {
            "bootstrap_rmse_mean": rmse_mean,
            "bootstrap_rmse_std": rmse_std,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "n_bootstraps": n_bootstraps,
            "description": "Bootstrap confidence intervals represent the range in which the true RMSE is likely to fall 95% of the time if we were to resample the test data. A narrow CI indicates high model stability."
        }
        
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"\nResults saved to {OUTPUT_JSON}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        spark.stop()

if __name__ == "__main__":
    run_analysis()
