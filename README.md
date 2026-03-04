# NYC Yellow Taxi Trip Dataset 2023 - Dual-Stage ML Pipeline
**Author**: Chinedum Okpechi
**Module**: 7006SCN - Machine Learning and Big Data

---
## 🚨 IMPORTANT: Main Project Report 🚨
**Please read the comprehensive final project report first:**  
[👉 **`reports/Fare prediction - Chinedum Okpechi 16621833 - 7006SCN - Machine Learning and Big Data..pdf`**](reports/Fare%20prediction%20-%20Chinedum%20Okpechi%2016621833%20-%207006SCN%20-%20Machine%20Learning%20and%20Big%20Data..pdf)

*(This report contains the full breakdown of the Medallion Architecture, Dual-Stage ML Pipeline, clustering characteristics, evaluation metrics, and business insights.)*
---

This project implements a comprehensive, end-to-end Big Data and Machine Learning pipeline analyzing the New York City Yellow Taxi Trip Dataset from 2023. The pipeline features a Medallion Architecture for data processing and a dual-stage machine learning approach combining unsupervised clustering and supervised regression.

## Project Architecture

### 1. Medallion Data Architecture (PySpark)
The data processing pipeline is built using Apache Spark (PySpark) to handle large-scale data efficiently:
- **Bronze Layer (`scripts/bronze_ingestion.py`)**: Raw data ingestion into Parquet format. Includes the ingestion of taxi zones for spatial joins.
- **Silver Layer (`scripts/silver_processing.py`)**: Data cleaning (handling nulls and duplicates) and feature engineering (time-based features, one-hot encoding). Implements an optimized **Broadcast Join** mapping pickup location IDs to actual boroughs and zones.
- **Gold Layer (`scripts/gold_training_export.py`)**: Final data preparation, scaling, and feature selection for Machine Learning.

### 2. Dual-Stage Machine Learning Pipeline
The project utilizes a two-stage approach to extract insights and predict outcomes:
- **Stage 1 (Unsupervised Learning - Clustering)**: Discovering passenger personas and trip patterns.
  - Advanced K-Means (`clustering_kmeans_advanced.py`)
  - Bisecting K-Means (`clustering_bisecting_kmeans.py`)
  - Gaussian Mixture Models (GMM) (`clustering_gmm.py`)
  - GPU-Accelerated PyTorch Clustering (`clustering_gpu_pytorch.py`)
  - Comprehensive clustering evaluation and stability testing.
- **Stage 2 (Supervised Learning - Regression)**: Fare Prediction models.
  - Model training utilizing optimized features from the Gold layer (`model_training.py`, `model_training_sklearn.py`).

### 3. Business Intelligence & Dashboards
The pipeline integrates with **Tableau** to provide actionable business dashboards. Cleaned and aggregated metrics are exported to the `tableau/data` directory for seamless dashboard rendering (e.g., `dashboard_model_performance.csv`).

## Directory Structure
- `data/`: Contains Bronze, Silver, and Gold Parquet datasets.
- `scripts/`: Python and PySpark scripts for data processing and model training.
- `models/`: Serialized pre-trained ML models.
- `results/` & `reports/`: Exported evaluation metrics, predictions, cluster details, and the final project report.
- `tableau/`: Data exports used for Tableau visualization.
- `images/`: Generated plots and diagrams (e.g., LIME interpretability, cluster scatters, RMSE analysis).

## How to Run

### Run Full Pipeline
To orchestrate the complete analytical pipeline from data ingestion to model training, execute the main run script:
```bash
python scripts/run_pipeline.py
```

### Custom Fare Prediction
A dedicated shell script is provided to test custom fare predictions on the trained model.
```bash
./run_prediction.sh
```
_(Alternatively, run `python scripts/run_spark_with_conda.py scripts/predict_custom_fare.py` directly)_

---
*Developed for 7006SCN Coursework Requirements.*
