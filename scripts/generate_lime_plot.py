import os
import lime
import lime.lime_tabular
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler

def generate_lime():
    print("Initializing environment for LIME Explanation generation...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gold_data_path = os.path.join(base_dir, "data", "gold", "hourly_stats.parquet")  # Used for visualization references
    silver_data_path = os.path.join(base_dir, "data", "silver", "taxi_features.parquet")
    model_path = os.path.join(base_dir, "models", "mllib_best_model")
    results_dir = os.path.join(base_dir, "results")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    spark = SparkSession.builder \
        .appName("LIME_Generator") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    try:
        print("Loading Best Model...")
        model = PipelineModel.load(model_path)
        
        print("Loading Silver testing data...")
        df = spark.read.parquet(silver_data_path)
        
        # We need a small sample for LIME
        sample_df = df.sample(fraction=0.01, seed=42).limit(100)
        
        features = [
             'trip_distance', 'pickup_hour', 'day_of_week', 'is_weekend', 'passenger_count',
             'trip_duration_min',
             'VendorID_1', 'VendorID_2', 
             'payment_type_1', 'payment_type_2', 'payment_type_3', 'payment_type_4', 'payment_type_5',
             'store_and_fwd_flag_N', 'store_and_fwd_flag_Y',
             'RatecodeID_1', 'RatecodeID_2', 'RatecodeID_3', 'RatecodeID_4', 'RatecodeID_5', 'RatecodeID_6', 'RatecodeID_99'
        ]
        
        available = set(sample_df.columns)
        final_features = [f for f in features if f in available]

        print("Extracting feature arrays to localized Pandas/Numpy context for LIME...")
        pandas_df = sample_df.select(final_features).toPandas()
        labels_df = sample_df.select("fare_amount").toPandas()
        
        X_numpy = pandas_df.values
        y_numpy = labels_df.values.flatten()
        
        # Wrapper function to allow LIME to pass localized Numpy arrays back through to PySpark pipeline
        def predict_fn(data_array):
            # Convert the perturbed numpy arrays back into Spark DataFrame
            import pandas as pd
            temp_pdf = pd.DataFrame(data_array, columns=final_features)
            # Add dummy labels as Pipeline requires them
            temp_pdf['fare_amount'] = 0.0
            temp_sdf = spark.createDataFrame(temp_pdf)
            
            # Run inference
            predictions = model.transform(temp_sdf)
            preds_pd = predictions.select("prediction").toPandas()
            return preds_pd.values.flatten()
            
        print("Initializing LIME Tabular Explainer...")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_numpy,
            feature_names=final_features,
            class_names=['fare_amount'],
            mode='regression',
            random_state=42
        )
        
        print("Explaining observation 0 (simulated High-Value Trip) ...")
        # Attempt to find a high value trip in sample
        high_value_idx = np.argmax(y_numpy)
        print(f"Explaining trip at index {high_value_idx} with actual fare: ${y_numpy[high_value_idx]:.2f}")
        
        exp = explainer.explain_instance(
            data_row=X_numpy[high_value_idx], 
            predict_fn=predict_fn,
            num_features=10
        )
        
        html_path = os.path.join(results_dir, "lime_explanation.html")
        print(f"Saving LIME explanation visual to {html_path}")
        exp.save_to_file(html_path)
        
        print("LIME Generation Complete.")

    except Exception as e:
        print(f"Error executing LIME generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()

if __name__ == "__main__":
    generate_lime()
