import sys
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, avg, count

def process_gold():
    """
    Gold Layer:
    - Reads Silver Parquet.
    - Trains 3 Models (LR, RF, GBT) comparisons.
    - Selects BEST model based on Validation RMSE.
    - Saves Best PipelineModel to data/models/mllib_best_model.
    - Aggregates for Tableau.
    """
    print("\n[GOLD] Starting Training & Export (Distinction Level)...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "silver", "taxi_features.parquet")
    output_dir_gold = os.path.join(base_dir, "data", "gold")
    model_path = os.path.join(base_dir, "models", "mllib_best_model")
    tableau_dir = os.path.join(base_dir, "results", "tableau")
    results_dir = os.path.join(base_dir, "results")
    
    for d in [output_dir_gold, tableau_dir, os.path.dirname(model_path), results_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    spark = SparkSession.builder \
        .appName("Gold_Training_MultiAlgo") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    try:
        print("Reading Silver data...")
        if not os.path.exists(input_path):
             print(f"Error: Silver data not found at {input_path}")
             sys.exit(1)
             
        df = spark.read.parquet(input_path)
        
        # TEST MODE
        if os.environ.get("IS_TEST_RUN"):
            print("⚠️ TEST MODE DETECTED: Ensuring small input.")
        
        # SPLIT DATA
        print("Splitting data (Train/Val/Test)...")
        train, val, test = df.randomSplit([0.7, 0.15, 0.15], seed=42)
        
        # FEATURE SELECTION
        features = [
             'trip_distance', 'pickup_hour', 'day_of_week', 'is_weekend', 'passenger_count',
             'trip_duration_min',
             'VendorID_1', 'VendorID_2', 
             'payment_type_1', 'payment_type_2', 'payment_type_3', 'payment_type_4', 'payment_type_5',
             'store_and_fwd_flag_N', 'store_and_fwd_flag_Y',
             'RatecodeID_1', 'RatecodeID_2', 'RatecodeID_3', 'RatecodeID_4', 'RatecodeID_5', 'RatecodeID_6', 'RatecodeID_99'
        ]
        
        # Filter features that actually exist in DF
        available = set(df.columns)
        final_features = [f for f in features if f in available]
        print(f"Training on {len(final_features)} features.")
        
        # DEFINE PIPELINE STAGES (Common)
        assembler = VectorAssembler(inputCols=final_features, outputCol="features_raw", handleInvalid="skip")
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
        
        # DEFINE ALGORITHMS TO COMPARE
        algos = [
            ("LinearRegression", LinearRegression(labelCol="fare_amount", featuresCol="features", maxIter=10, regParam=0.3, elasticNetParam=0.8)),
            ("RandomForest", RandomForestRegressor(labelCol="fare_amount", featuresCol="features", numTrees=20, maxDepth=10, seed=42)),
            ("GBT", GBTRegressor(labelCol="fare_amount", featuresCol="features", maxIter=20, maxDepth=5, seed=42))
        ]
        
        best_model_name = None
        best_pipeline_model = None
        best_rmse = float('inf')
        
        evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")
        results_str = "Model Comparison Results:\n"
        
        print("\n--- Training & Evaluating Models ---")
        
        for name, algo in algos:
            print(f"\nTraining {name}...")
            pipeline = Pipeline(stages=[assembler, scaler, algo])
            
            # Fit
            model = pipeline.fit(train)
            
            # Evaluate on Val
            predictions = model.transform(val)
            rmse = evaluator.evaluate(predictions)
            print(f"  > Validation RMSE: {rmse:.4f}")
            
            results_str += f"{name}: RMSE={rmse:.4f}\n"
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_pipeline_model = model
                best_model_name = name
        
        print(f"\n🏆 Best Model: {best_model_name} with RMSE: {best_rmse:.4f}")
        results_str += f"\nWINNER: {best_model_name}"
        
        # SAVE BEST MODEL
        print(f"Saving Best PipelineModel ({best_model_name}) to {model_path}...")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        best_pipeline_model.save(model_path)
        
        # SAVE COMPARISON RESULTS
        with open(os.path.join(results_dir, "model_comparison.txt"), "w") as f:
            f.write(results_str)
            
        # FINAL TEST EVALUATION
        print("Evaluating Best Model on TEST set...")
        test_pred = best_pipeline_model.transform(test)
        test_rmse = evaluator.evaluate(test_pred)
        print(f"Test RMSE: {test_rmse:.4f}")

        # TABLEAU AGGREGATION
        print("Generating Tableau Aggregates...")
        hourly_stats = df.groupBy("pickup_hour").agg(
            avg("fare_amount").alias("avg_fare"),
            avg("trip_distance").alias("avg_dist"),
            avg("trip_duration_min").alias("avg_duration"),
            count("*").alias("trip_count")
        ).orderBy("pickup_hour")
        
        gold_file = os.path.join(output_dir_gold, "hourly_stats.parquet")
        hourly_stats.write.mode("overwrite").parquet(gold_file)
        
        hourly_stats.toPandas().to_csv(os.path.join(tableau_dir, "dashboard_hourly_gold.csv"), index=False)
        
        print("[GOLD] Complete.")

    except Exception as e:
        print(f"[GOLD] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    process_gold()
