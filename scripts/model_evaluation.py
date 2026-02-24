import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import os
import sys
from pyspark.sql import SparkSession

def evaluate_model():
    """
    Evaluates the Random Forest model using Cross-Validation and a Hold-out Test Set.
    Reads data using native PySpark DataFrames instead of Pandas.
    Generates plots and metrics.
    """
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "train.parquet")
    test_path = os.path.join(base_dir, "data", "processed", "test.parquet")
    output_dir = os.path.join(base_dir, "results", "plots")
    metrics_path = os.path.join(base_dir, "results", "metrics.txt")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Train or Test file not found.")
        sys.exit(1)
        
    print("Initializing Model Evaluation via PySpark Loader...")
    
    spark = SparkSession.builder \
        .appName("Model_Evaluation_Loader") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()
        
    try:
        # Features & Target
        leakage_cols = [
            'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'fare_amount', 'total_amount',
            'tip_amount', 'tolls_amount', 'mta_tax', 'extra',
            'improvement_surcharge', 'congestion_surcharge', 'airport_fee',
            'PULocationID', 'DOLocationID'
        ]
        
        # 1. Load Data
        print("Loading 5% Training Sample for CV via PySpark...")
        train_df = spark.read.parquet(train_path)
        train_sample_df = train_df.sample(fraction=0.05, seed=42)
        
        all_cols = train_df.columns
        feature_cols = [c for c in all_cols if c not in leakage_cols]
        
        print("Collecting Train sample to Driver...")
        train_rows = train_sample_df.select(*(feature_cols + ['fare_amount'])).collect()
        
        X_train = np.array([[row[f] for f in feature_cols] for row in train_rows], dtype=np.float32)
        y_train = np.array([row['fare_amount'] for row in train_rows], dtype=np.float32)
        
        print(f"Training Sample Shape: {X_train.shape}")
        
        # 2. Cross-Validation (5-Fold)
        print("\n" + "="*50)
        print("STEP 1: 5-FOLD CROSS-VALIDATION")
        print("="*50)
        
        rf = RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs=-1, random_state=42)
        
        cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        cv_rmse = -cv_scores
        
        print(f"CV RMSE Scores: {cv_rmse}")
        print(f"Mean CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
        
        # 3. Train Final Model
        print("\n" + "="*50)
        print("STEP 2: TRAINING FINAL MODEL & TEST EVALUATION")
        print("="*50)
        
        rf.fit(X_train, y_train)
        
        # Load Test Set via PySpark
        print("Loading Full Test Set via PySpark...")
        test_df = spark.read.parquet(test_path)
        
        # For Test Set, if it's 5.5M rows, predicting locally over everything might be large memory.
        # But evaluation is needed, so let's collect it. 
        print("Collecting Test Set to Driver...")
        test_rows = test_df.select(*(feature_cols + ['fare_amount'])).collect()
        
        X_test = np.array([[row[f] for f in feature_cols] for row in test_rows], dtype=np.float32)
        y_test = np.array([row['fare_amount'] for row in test_rows], dtype=np.float32)
        
        print("Predicting on Test Set...")
        y_pred = rf.predict(X_test)
        
        # Metrics
        final_rmse = root_mean_squared_error(y_test, y_pred)
        final_mae = mean_absolute_error(y_test, y_pred)
        print("Calculating R2 Score...")
        final_r2 = r2_score(y_test, y_pred)
        
        print(f"Test RMSE: {final_rmse:.4f}")
        print(f"Test MAE:  {final_mae:.4f}")
        print(f"Test R2:   {final_r2:.4f}")
        
        # Save Metrics
        with open(metrics_path, "w") as f:
            f.write("Model Evaluation Metrics\n")
            f.write("========================\n")
            f.write(f"CV RMSE (Mean): {cv_rmse.mean():.4f}\n")
            f.write(f"Test RMSE:      {final_rmse:.4f}\n")
            f.write(f"Test MAE:       {final_mae:.4f}\n")
            f.write(f"Test R2:        {final_r2:.4f}\n")

        # 4. Visualizations
        print("\nGenerating Plots...")
        sns.set_theme(style="whitegrid")
        
        # Feature Importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = 10
        
        plt.figure(figsize=(10, 6))
        plt.title("Top 10 Feature Importances")
        plt.bar(range(top_n), importances[indices[:top_n]], align="center")
        plt.xticks(range(top_n), [feature_cols[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"))
        plt.close()
        
        # Pred vs Actual (Sampled to avoid dense blob)
        plot_indices = np.random.choice(len(y_test), 10000, replace=False)
        y_test_sample = y_test[plot_indices]
        y_pred_sample = y_pred[plot_indices]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test_sample, y_pred_sample, alpha=0.3, s=10)
        
        min_val = min(y_test_sample.min(), y_pred_sample.min())
        max_val = max(y_test_sample.max(), y_pred_sample.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.title("Predicted vs Actual (Test Sample)")
        plt.xlabel("Actual Fare (Scaled)")
        plt.ylabel("Predicted Fare (Scaled)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pred_vs_actual.png"))
        plt.close()
        
        print(f"\nEvaluation Complete. Metrics saved to {metrics_path}. Plots saved to {output_dir}")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    evaluate_model()
