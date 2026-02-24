import os
import sys
import numpy as np
from sklearn.cluster import KMeans
import time
from pyspark.sql import SparkSession

def run_stability_test():
    """
    Performs 'Thorough stability testing with statistical validation' 
    by repeatedly resampling the data (Bootstrapping) and measuring 
    how much the cluster centroids shift.
    Reads data via PySpark.
    """
    print("\n[STABILITY] Starting Bootstrap Stability Test for Clustering...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "silver", "taxi_features.parquet")
    results_dir = os.path.join(base_dir, "results")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    start_time = time.time()
    
    spark = SparkSession.builder \
        .appName("Clustering_Stability_Test") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
        
    try:
        df = spark.read.parquet(input_path)
        total_rows = df.count()
        sample_size = min(total_rows, 100000)
        
        fraction = sample_size / total_rows
        df_sample = df.sample(withReplacement=False, fraction=fraction, seed=42).limit(sample_size)
        
        features = ['trip_distance', 'fare_amount', 'trip_duration_min']
        
        print("  Collecting data from PySpark...")
        rows = df_sample.select(*features).collect()
        
        X_raw = np.array([[row[f] for f in features] for row in rows])
        
        # Standardize
        X_mean = np.mean(X_raw, axis=0)
        X_std = np.std(X_raw, axis=0)
        X_std[X_std == 0] = 1.0 # prevent div by zero
        X = (X_raw - X_mean) / X_std
        
        k = 4
        n_bootstraps = 10 
        
        print(f"  Running {n_bootstraps} bootstraps on {sample_size} records to test cluster stability...")
        
        # Baseline model
        base_kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        base_kmeans.fit(X)
        base_centroids = np.sort(base_kmeans.cluster_centers_, axis=0)
        
        centroid_shifts = []
        
        for i in range(n_bootstraps):
            # Sample with replacement
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx]
            
            boot_kmeans = KMeans(n_clusters=k, random_state=np.random.randint(1000), n_init=5)
            boot_kmeans.fit(X_boot)
            
            # Sort centroids to align them roughly for comparison
            boot_centroids = np.sort(boot_kmeans.cluster_centers_, axis=0)
            
            # Calculate mean shift
            shift = np.mean(np.linalg.norm(base_centroids - boot_centroids, axis=1))
            centroid_shifts.append(shift)
            
            print(f"    Bootstrap {i+1}/{n_bootstraps} | Avg Centroid Shift: {shift:.4f}")
            
        avg_shift = np.mean(centroid_shifts)
        std_shift = np.std(centroid_shifts)
        
        duration = time.time() - start_time
        
        results_str = "--- Cluster Stability Test (Bootstrapping) ---\n"
        results_str += f"Total Time: {duration:.2f}s\n"
        results_str += f"Bootstraps: {n_bootstraps}\n"
        results_str += f"Clusters (K): {k}\n"
        results_str += f"Mean Centroid Shift: {avg_shift:.6f} (Standardized units)\n"
        results_str += f"Shift Std Dev: {std_shift:.6f}\n\n"
        
        if avg_shift < 0.1:
            results_str += "CONCLUSION: The cluster assignments are highly stable.\n"
        else:
            results_str += "CONCLUSION: The clusters show structural instability.\n"
            
        print("\n" + results_str)
        
        with open(os.path.join(results_dir, "clustering_stability_results.txt"), "w") as f:
            f.write(results_str)
            
    finally:
        spark.stop()

if __name__ == "__main__":
    run_stability_test()
