# Tableau Dashboards Guide

This directory contains the necessary structure and exported data to build the 4 Tableau Dashboards dictated by the 7006SCN Coursework Brief.

## Data Sources
The underlying Parquet architecture outputs specialized CSV flat-files optimized for Tableau ingestion. These are located in `./data/`.

* `dashboard_data_quality.csv`
* `dashboard_business_hour.csv` 
* `dashboard_distance_distribution.csv`
* `dashboard_feature_importance.csv`
* `dashboard_cluster_profiles.csv`
* `dashboard_scaling_strong.csv` // etc..

---

## Required Dashboard Mapping (Student Task)
You must import the above CSV files into Tableau Public/Desktop to build the `.twbx` files. Save your workbooks in this root `tableau/` directory.

### Dashboard 1: Data quality and pipeline monitoring
**Required Files:** `dashboard_data_quality.csv`
**Objective:** Visualize null counts, outlier clipping bounds, and volume reductions through the Bronze -> Silver tier progression.

### Dashboard 2: Model performance and feature importance
**Required Files:** `dashboard_cluster_profiles.csv`, `dashboard_feature_importance.csv`, `dashboard_model_performance.csv`
**Objective:** Present the Silhouette/Davies-Bouldin stability metrics for the Mobility Personas, alongside the GBT regression feature importance chart (Trip Distance vs Fare).

### Dashboard 3: Business insights and recommendations
**Required Files:** `dashboard_business_hour.csv`, `dashboard_distance_distribution.csv`, `dashboard_predictions.csv`
**Objective:** Present the actionable insights. Map fare distributions against time of day and show how different Mobility Personas drive distinct revenue streams.

### Dashboard 4: Scalability and cost analysis
**Required Files:** `dashboard_scaling_strong.csv`, `dashboard_scaling_weak.csv`, `dashboard_cost_performance.csv`
**Objective:** Graph PySpark Spark Partition Scaling vs execution time, and compare them against the PyTorch MPS Apple Silicon GPU acceleration times.
