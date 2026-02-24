import os
import sys
import argparse
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import lit

def predict_custom_fare():
    parser = argparse.ArgumentParser(description="Predict Taxi Fare")
    parser.add_argument("--distance", type=float, default=20.0, help="Trip distance in miles")
    parser.add_argument("--duration", type=float, default=10.0, help="Trip duration in minutes")
    parser.add_argument("--passengers", type=float, default=2.0, help="Number of passengers")
    
    # Handle sys.argv shifting from the wrapper script
    cli_args = sys.argv[1:]
    if len(cli_args) > 0 and cli_args[0].endswith('.py'):
        cli_args = cli_args[1:]
        
    args, unknown = parser.parse_known_args(cli_args)

    print("\n--- Custom Fare Prediction ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "mllib_best_model")
    input_path = os.path.join(base_dir, "data", "silver", "taxi_features.parquet")
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}.")
        print("Please run the pipeline first.")
        sys.exit(1)
        
    spark = SparkSession.builder \
        .appName("FarePrediction") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        # Load the trained PipelineModel
        print("Loading trained model...")
        model = PipelineModel.load(model_path)
        
        # Load ONE row from the silver dataset just to get the exact schema
        print("Loading baseline schema...")
        df = spark.read.parquet(input_path).limit(1)
        
        print(f"Injecting custom values: {args.duration} minutes, {args.passengers} passengers, {args.distance} miles...")
        custom_df = df.withColumn("trip_duration_min", lit(args.duration)) \
                      .withColumn("passenger_count", lit(args.passengers)) \
                      .withColumn("trip_distance", lit(args.distance))
        
        # Run prediction
        print("Running inference...")
        predictions = model.transform(custom_df)
        
        # Extract and print result
        fare_prediction = predictions.select("prediction").collect()[0]["prediction"]
        
        print("\n" + "="*50)
        print(f"🚕 PREDICTED FARE: ${fare_prediction:.2f}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    predict_custom_fare()
