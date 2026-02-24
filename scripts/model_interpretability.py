import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os
import sys
from pyspark.sql import SparkSession

def explain_model():
    """
    Generates SHAP values and plots to explain the Random Forest model.
    Reads data using native PySpark DataFrames instead of Pandas.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "production_rf_model.joblib")
    test_path = os.path.join(base_dir, "data", "processed", "test.parquet")
    output_dir = os.path.join(base_dir, "results", "plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)
        
    print("Loading Production Model...")
    rf_model = joblib.load(model_path)
    
    print("Loading Test Sample via PySpark Loader...")
    
    spark = SparkSession.builder \
        .appName("Model_Interpretability_Loader") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
        
    try:
        test_df = spark.read.parquet(test_path)
        
        # Load small sample for SHAP (computationally expensive)
        total_test_rows = test_df.count()
        fraction = 500 / total_test_rows if total_test_rows > 0 else 1.0
        test_sample_df = test_df.sample(withReplacement=False, fraction=fraction, seed=42).limit(500)
        
        # Features (Exclude Leakage)
        leakage_cols = [
            'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'fare_amount', 'total_amount',
            'tip_amount', 'tolls_amount', 'mta_tax', 'extra',
            'improvement_surcharge', 'congestion_surcharge', 'airport_fee',
            'PULocationID', 'DOLocationID'
        ]
        
        all_cols = test_sample_df.columns
        feature_cols = [c for c in all_cols if c not in leakage_cols]
        
        print(f"Collecting {test_sample_df.count()} samples...")
        rows = test_sample_df.select(*feature_cols).collect()
        
        X_test = np.array([[row[f] for f in feature_cols] for row in rows], dtype=np.float32)
        
        print(f"Explaining predictions for {X_test.shape[0]} samples...")
        print("Initializing SHAP TreeExplainer...")
        
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)
        
        print("Generating SHAP Summary Plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
        
        output_path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP Summary Plot saved to {output_path}")

    finally:
        spark.stop()

if __name__ == "__main__":
    explain_model()
