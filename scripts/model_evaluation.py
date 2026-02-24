import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import os
import sys

def evaluate_model():
    """
    Evaluates the Random Forest model using Cross-Validation and a Hold-out Test Set.
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
        
    print("Initializing Model Evaluation...")
    
    # Features & Target
    leakage_cols = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        'fare_amount', 'total_amount',
        'tip_amount', 'tolls_amount', 'mta_tax', 'extra',
        'improvement_surcharge', 'congestion_surcharge', 'airport_fee',
        'PULocationID', 'DOLocationID'
    ]
    
    # 1. Load Data
    print("Loading 5% Training Sample for CV...")
    train_sample = pd.read_parquet(train_path).sample(frac=0.05, random_state=42)
    
    all_cols = train_sample.columns.tolist()
    feature_cols = [c for c in all_cols if c not in leakage_cols]
    
    X_train = train_sample[feature_cols]
    y_train = train_sample['fare_amount']
    
    print(f"Training Sample Shape: {X_train.shape}")
    
    # 2. Cross-Validation (5-Fold)
    print("\n" + "="*50)
    print("STEP 1: 5-FOLD CROSS-VALIDATION")
    print("="*50)
    
    rf = RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs=-1, random_state=42)
    
    # scoring='neg_root_mean_squared_error' returns negative values
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    cv_rmse = -cv_scores
    
    print(f"CV RMSE Scores: {cv_rmse}")
    print(f"Mean CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
    
    # 3. Train Final Model
    print("\n" + "="*50)
    print("STEP 2: TRAINING FINAL MODEL & TEST EVALUATION")
    print("="*50)
    
    rf.fit(X_train, y_train)
    
    # Load Test SEt
    print("Loading Full Test Set (5.5M rows)...")
    test_df = pd.read_parquet(test_path)
    X_test = test_df[feature_cols]
    y_test = test_df['fare_amount']
    
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
    # Sample 10,000 points for the plot
    plot_indices = np.random.choice(len(y_test), 10000, replace=False)
    y_test_sample = y_test.iloc[plot_indices]
    y_pred_sample = y_pred[plot_indices]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_sample, y_pred_sample, alpha=0.3, s=10)
    
    # Perfect calibration line
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

if __name__ == "__main__":
    evaluate_model()
