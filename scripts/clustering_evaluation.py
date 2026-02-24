import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import time

def evaluate_clusters(file_path, model_name):
    """
    Evaluates clustered Parquet files using 5 Metrics:
    1. Silhouette Score 
    2. WSSSE (Inertia proxy)
    3. Calinski-Harabasz Index
    4. Davies-Bouldin Index
    5. Cluster Balance (Entropy)
    """
    print(f"\n[EVAL] Evaluating format: {model_name} from {os.path.basename(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
        
    start_time = time.time()
    
    # Read data. To prevent OOM on single node during evaluation, we use a stratified sample if needed 
    # but 50k - 500k rows is easily handled by scikit-learn.
    df = pd.read_parquet(file_path)
    
    # Subsample for very slow metrics like Silhouette if dataset is massive (>100k)
    sample_size = min(len(df), 50000)
    if sample_size < len(df):
        print(f"  Subsampling {sample_size} rows for metric calculation...")
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
        
    features = ['trip_distance', 'fare_amount', 'trip_duration_min']
    X = df_sample[features].values
    labels = df_sample['prediction'].values
    
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        print("[EVAL] Error: Need at least 2 clusters for evaluation.")
        return
        
    print("  Calculating Silhouette Score...")
    silhouette = silhouette_score(X, labels)
    
    print("  Calculating Calinski-Harabasz Index...")
    ch_score = calinski_harabasz_score(X, labels)
    
    print("  Calculating Davies-Bouldin Index...")
    db_score = davies_bouldin_score(X, labels)
    
    print("  Calculating WSSSE proxy (Inertia)...")
    # WSSSE manually: sum of squared distances to centroid
    wssse = 0
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        centroid = cluster_points.mean(axis=0)
        wssse += np.sum((cluster_points - centroid)**2)
        
    print("  Calculating Cluster Balance (Entropy)...")
    # Entropy: -sum(p * log(p)) where p is proportion in cluster
    counts = np.bincount(labels)
    probs = counts[counts > 0] / len(labels)
    entropy = -np.sum(probs * np.log(probs))
    
    duration = time.time() - start_time
    
    results_str = f"--- {model_name} Evaluation ---\n"
    results_str += f"Eval Time: {duration:.2f}s (Sample size: {sample_size})\n"
    results_str += f"Clusters (K): {n_clusters}\n"
    results_str += f"1. Silhouette Score: {silhouette:.4f} (Higher is better, cohesion/separation)\n"
    results_str += f"2. Calinski-Harabasz: {ch_score:.4f} (Higher is better, variance ratio)\n"
    results_str += f"3. Davies-Bouldin: {db_score:.4f} (Lower is better, cluster similarity)\n"
    results_str += f"4. WSSSE: {wssse:.4f} (Lower is better, compactness)\n"
    results_str += f"5. Cluster Entropy: {entropy:.4f} (Balances cluster sizes)\n\n"
    
    for k, p, c in zip(np.unique(labels), probs, counts[counts > 0]):
        results_str += f"   Cluster {k} Size: {c} ({p*100:.1f}%)\n"
        
    print(results_str)
    
    # Save append
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    with open(os.path.join(results_dir, "clustering_evaluation_metrics.txt"), "a") as f:
        f.write(results_str + "\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gold_dir = os.path.join(base_dir, "data", "gold")
    
    # Clear the file first
    results_file = os.path.join(base_dir, "results", "clustering_evaluation_metrics.txt")
    if os.path.exists(results_file):
        os.remove(results_file)
        
    # Evaluate whatever clusters we have
    evaluate_clusters(os.path.join(gold_dir, "clustered_trips_kmeans.parquet"), "Advanced K-Means (Spark)")
    evaluate_clusters(os.path.join(gold_dir, "clustered_trips_bisecting.parquet"), "Bisecting K-Means (Spark)")
    evaluate_clusters(os.path.join(gold_dir, "clustered_trips_gmm.parquet"), "Gaussian Mixture Model (Spark)")
