import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def train_models_sklearn():
    """
    Trains models using Scikit-Learn.
    Loads data using PySpark instead of Pandas.
    """
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "train.parquet")
    val_path = os.path.join(base_dir, "data", "processed", "validation.parquet")
    
    if not os.path.exists(train_path):
        print(f"Error: Train file {train_path} not found.")
        sys.exit(1)
        
    print("Initializing Sklearn Training via PySpark Loader...")
    
    spark = SparkSession.builder \
        .appName("Sklearn_Training_Loader") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()
        
    # 1. Define Features & Target
    leakage_cols = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        'fare_amount', 'total_amount',
        'tip_amount', 'tolls_amount', 'mta_tax', 'extra',
        'improvement_surcharge', 'congestion_surcharge', 'airport_fee',
        'PULocationID', 'DOLocationID'
    ]
    
    try:
        print("Loading Validation set...")
        val_df = spark.read.parquet(val_path)
        
        all_cols = val_df.columns
        feature_cols = [c for c in all_cols if c not in leakage_cols]
        
        # Collect to driver array
        print("Collecting memory to NumPy Arrays...")
        val_rows = val_df.select(*(feature_cols + ['fare_amount'])).collect()
        
        # Create NumPy arrays
        X_val = np.array([[row[f] for f in feature_cols] for row in val_rows], dtype=np.float32)
        y_val = np.array([row['fare_amount'] for row in val_rows], dtype=np.float32)
        
        print(f"Validation Shape: {X_val.shape}")
        
        # ---------------------------------------------------------
        # MODEL 1: LINEAR REGRESSION (SGD)
        # ---------------------------------------------------------
        print("\n" + "="*50)
        print("TRAINING MODEL 1: SGD Linear Regressor (Incremental via PySpark Parts)")
        print("="*50)
        
        sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
        
        train_df = spark.read.parquet(train_path)
        
        # Use simple collect for training chunk since SGD incremental is slow with massive loops over small row collections
        # Better to train on a large sampled subset for demonstration, or collect and partial fit.
        # We will train on a 10% sample for SGD.
        print("Taking a 10% sample of Train set for SGD...")
        sgd_train_df = train_df.sample(fraction=0.1, seed=42)
        sgd_rows = sgd_train_df.select(*(feature_cols + ['fare_amount'])).collect()
        
        X_train_sgd = np.array([[row[f] for f in feature_cols] for row in sgd_rows], dtype=np.float32)
        y_train_sgd = np.array([row['fare_amount'] for row in sgd_rows], dtype=np.float32)
        
        sgd.fit(X_train_sgd, y_train_sgd)
        print("SGD Training Complete.")
        
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
        
        rf = RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs=-1, random_state=42)
        
        print("Loading 5% sample for RF...")
        rf_train_df = train_df.sample(fraction=0.05, seed=42)
        rf_rows = rf_train_df.select(*(feature_cols + ['fare_amount'])).collect()
        
        X_train_rf = np.array([[row[f] for f in feature_cols] for row in rf_rows], dtype=np.float32)
        y_train_rf = np.array([row['fare_amount'] for row in rf_rows], dtype=np.float32)
        
        print(f"RF Training Data Shape: {X_train_rf.shape}")
        
        rf.fit(X_train_rf, y_train_rf)
        print("RF Training Complete.")
        
        # Evaluate RF
        y_pred_rf = rf.predict(X_val)
        rmse_rf = root_mean_squared_error(y_val, y_pred_rf)
        print(f"Random Forest Validation RMSE: {rmse_rf:.4f}")

    finally:
        spark.stop()

if __name__ == "__main__":
    train_models_sklearn()
