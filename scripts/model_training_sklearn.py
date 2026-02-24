import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import os
import sys
import pyarrow.parquet as pq

def train_models_sklearn():
    """
    Trains models using Scikit-Learn (Fallback for PySpark).
    1. SGDRegressor (Linear) - Trained incrementally.
    2. Random Forest - Trained on a subset (1M rows).
    """
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "train.parquet")
    val_path = os.path.join(base_dir, "data", "processed", "validation.parquet")
    
    if not os.path.exists(train_path):
        print(f"Error: Train file {train_path} not found.")
        sys.exit(1)
        
    print("Initializing Sklearn Training...")
    
    # 1. Define Features & Target
    # We must explicitly exclude leakage columns
    leakage_cols = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        'fare_amount', 'total_amount',
        'tip_amount', 'tolls_amount', 'mta_tax', 'extra',
        'improvement_surcharge', 'congestion_surcharge', 'airport_fee',
        'PULocationID', 'DOLocationID' # Exclude large categorical IDs
    ]
    
    # Validation Data (Load fully - 5.5M rows is manageable for evaluation ~400MB)
    print("Loading Validation set...")
    val_df = pd.read_parquet(val_path)
    # Identify feature columns
    all_cols = val_df.columns.tolist()
    feature_cols = [c for c in all_cols if c not in leakage_cols]
    
    X_val = val_df[feature_cols]
    y_val = val_df['fare_amount'] # Target
    
    print(f"Validation Shape: {X_val.shape}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # ---------------------------------------------------------
    # MODEL 1: LINEAR REGRESSION (SGD)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("TRAINING MODEL 1: SGD Linear Regressor (Incremental)")
    print("="*50)
    
    sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    
    # Stream Train Check
    parquet_file = pq.ParquetFile(train_path)
    batch_size = 500_000
    
    total_train = 0
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
        df_chunk = batch.to_pandas()
        X_chunk = df_chunk[feature_cols]
        y_chunk = df_chunk['fare_amount']
        
        sgd.partial_fit(X_chunk, y_chunk)
        total_train += len(df_chunk)
        print(f"SGD: Processed {total_train} rows...", end='\r')
        
    print(f"\nSGD Training Complete on {total_train} rows.")
    
    # Evaluate SGD
    y_pred_sgd = sgd.predict(X_val)
    rmse_sgd = root_mean_squared_error(y_val, y_pred_sgd)
    print(f"SGD Linear Model Validation RMSE: {rmse_sgd:.4f}")
    
    # ---------------------------------------------------------
    # MODEL 2: RANDOM FOREST
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("TRAINING MODEL 2: Random Forest (Sampled 5%)")
    print("="*50)
    
    # Load 5% Sample from Train
    # We can use the existing 'split_data' logic or just sample during load
    # Here we just read a subset of row groups or use pandas sample
    # Reading 5% of 25M is ~1.25M rows.
    
    rf = RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs=-1, random_state=42)
    
    print("Loading 5% sample for RF...")
    train_sample = pd.read_parquet(train_path).sample(frac=0.05, random_state=42)
    X_train_sample = train_sample[feature_cols]
    y_train_sample = train_sample['fare_amount']
    
    print(f"RF Training Data Shape: {X_train_sample.shape}")
    
    rf.fit(X_train_sample, y_train_sample)
    print("RF Training Complete.")
    
    # Evaluate RF
    y_pred_rf = rf.predict(X_val)
    rmse_rf = root_mean_squared_error(y_val, y_pred_rf)
    print(f"Random Forest Validation RMSE: {rmse_rf:.4f}")

if __name__ == "__main__":
    train_models_sklearn()
