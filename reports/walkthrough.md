# Medallion Architecture Implementation & Optimization

I have successfully refactored your project into a **Medallion Architecture**, optimizing the data flow and model training pipeline for correctness and scalability.

## 🏗️ New Folder Structure

```text
data/
├── bronze/    # Raw Parquet (Ingested from CSV)
├── silver/    # Cleaned Parquet (Unscaled, Human-Readable Units)
└── gold/      # Aggregated Data & Model Artifacts
```

## 🚀 Pipeline Optimization (The "Optimal" Setup)

To ensure correct predictions and adhere to Big Data best practices, I implemented the following **Optimal Architecture**:

### 1. **Silver Layer (Business Logic)**
*   **Change**: Removed manual scaling (Z-Scores) from the Silver layer.
*   **Benefit**: Data in `data/silver/taxi_features.parquet` remains in **human-readable units** (miles, minutes). This makes debugging and analysis (e.g., Tableau) much easier.

### 2. **Gold Layer (Machine Learning Pipeline)**
*   **Change**: Wrapped the Preprocessing and Model into a single Spark ML `Pipeline`.
    *   `VectorAssembler` (Combines features)
    *   `StandardScaler` (Normalizes features automatically)
    *   `GBTRegressor` (Predicts Fare)
*   **Benefit**: The scaling logic is now **saved with the model**. This prevents "Training/Serving Skew" where the model expects scaled input but receives raw input (which caused the erroneous $61.79 prediction).

### 3. **Prediction**
*   **Change**: `predict_user_input.py` now loads the full `PipelineModel`.
*   **Benefit**: You pass raw inputs (e.g., "10 miles"), and the pipeline automatically scales them using the exact parameters from training.

### 3. Distinction-Grade Features (New!)
I have implemented the following advanced features to meet the "Distinction" criteria:

#### 🟢 Data Engineering: Broadcast Join
*   **What**: Ingested `taxi_zone_lookup.csv` and joined it with trip data using Spark's `broadcast()` optimization.
*   **Why**: Efficiently enriches massive trip table with `Borough` and `Zone` names without network shuffle.
*   **Code**: `scripts/ingest_zones.py`, `scripts/silver_processing.py`.

#### 🟢 Advanced ML: Multi-Algorithm Comparison
*   **What**: Automatically trains and compares **Linear Regression**, **Random Forest**, and **GBT Regressor**.
*   **Why**: Demonstrates model selection and hyperparameter tuning.
*   **Winner**: **RandomForest** (RMSE ~5.53).
*   **Code**: `scripts/gold_training_export.py`.

#### 🟢 Unsupervised Learning: Clustering
*   **What**: Implemented **K-Means Clustering** on trip behaviors (Distance, Fare, Duration).
*   **Why**: Identifies "Trip Personas" (e.g., Short-Cheap vs Long-Expensive).
*   **Code**: `scripts/clustering_kmeans.py`.

#### 🟢 Scalability Analysis
*   **What**: Benchmarked Single-Node (Scikit-Learn) vs Distributed (Spark) and Weak Scaling.
*   **Report**: See `scalability_report.md`.
*   **Code**: `scripts/sklearn_baseline.py`, `scripts/scalability_benchmark.py`.

## ✅ Verification Results

I ran the full pipeline in **Test Mode** (100k rows) and verified the prediction:

| Input | Old Model (Scaled) | **Optimized Model** |
| :--- | :--- | :--- |
| **20 min, 10 miles, 2 pax** | ~$61.79 (Incorrect) | **$38.71** (Realistic) |

## 🛠️ How to Run

### 1. Run Full Pipeline
To train on the full dataset (Recommended for final grade):
```bash
# Ensure IS_TEST_RUN is unset or false
unset IS_TEST_RUN
python scripts/run_pipeline.py
```

### 2. Run Prediction with Custom Inputs
```bash
python scripts/run_spark_with_conda.py predict_user_input.py
```
*(You can edit the `user_distance`, `user_duration` variables in the script)*
