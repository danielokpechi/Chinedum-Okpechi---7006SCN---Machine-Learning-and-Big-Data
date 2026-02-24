import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os
import sys

def explain_model():
    """
    Generates SHAP values and plots to explain the Random Forest model.
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
    
    print("Loading Test Sample (500 rows)...")
    # Load small sample for SHAP (computationally expensive)
    test_sample = pd.read_parquet(test_path).sample(n=500, random_state=42)
    
    # Features (Exclude Leakage - same logic as training)
    leakage_cols = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        'fare_amount', 'total_amount',
        'tip_amount', 'tolls_amount', 'mta_tax', 'extra',
        'improvement_surcharge', 'congestion_surcharge', 'airport_fee',
        'PULocationID', 'DOLocationID'
    ]
    
    all_cols = test_sample.columns.tolist()
    feature_cols = [c for c in all_cols if c not in leakage_cols]
    
    X_test = test_sample[feature_cols]
    
    print(f"Explaining predictions for {len(X_test)} samples...")
    print("Initializing SHAP TreeExplainer...")
    
    # TreeExplainer is optimized for Random Forests
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    
    output_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP Summary Plot saved to {output_path}")

if __name__ == "__main__":
    explain_model()
