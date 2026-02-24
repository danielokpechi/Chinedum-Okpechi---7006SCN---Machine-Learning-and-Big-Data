import os
import time
import torch
import pandas as pd
import numpy as np

def run_pytorch_kmeans():
    """
    Perform K-Means Clustering using PyTorch on Apple Metal GPU (MPS).
    """
    print("\n[GPU] Starting PyTorch K-Means Clustering on MPS...")
    
    # Check for MPS
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            sys.exit(1)
        else:
            print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            sys.exit(1)
            
    device = torch.device("mps")
    print(f"[GPU] Using device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "silver", "taxi_features.parquet")
    results_dir = os.path.join(base_dir, "results")
    
    try:
        # We read Parquet with Pandas for PyTorch ingestion
        print("[GPU] Loading dataset into memory...")
        # Subset for memory if test run
        if os.environ.get("IS_TEST_RUN"):
             df = pd.read_parquet(input_path).head(50000)
        else:
             df = pd.read_parquet(input_path)

        features = ['trip_distance', 'fare_amount', 'trip_duration_min']
        X_df = df[features].copy()
        
        # Standard Scaler
        print("[GPU] Normalizing data...")
        X_mean = X_df.mean()
        X_std = X_df.std()
        X_scaled = (X_df - X_mean) / X_std
        
        # Send to GPU
        print("[GPU] Moving data to Metal MPS...")
        X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32).to(device)
        
        # PyTorch K-Means implementation
        k = 5
        max_iter = 100
        tol = 1e-4
        n_samples, n_features = X_tensor.shape
        
        # Randomly initialize centroids
        torch.manual_seed(42)
        random_indices = torch.randperm(n_samples)[:k]
        centroids = X_tensor[random_indices].clone()
        
        print(f"[GPU] Training K-Means with k={k} on {n_samples} samples...")
        start_time = time.time()
        
        for i in range(max_iter):
            # Calculate Euclidean distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2<x,c>
            # Vectorized for GPU speed
            distances = torch.cdist(X_tensor, centroids, p=2.0)
            
            # Predict clusters
            cluster_assignments = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for j in range(k):
                mask = (cluster_assignments == j)
                if mask.sum() > 0:
                    new_centroids[j] = X_tensor[mask].mean(dim=0)
                else:
                    new_centroids[j] = centroids[j] # Keep old if empty
                    
            # Check convergence
            center_shift = torch.norm(centroids - new_centroids, p=2)
            centroids = new_centroids
            if center_shift < tol:
                print(f"[GPU] Converged at iteration {i}")
                break
                
        duration = time.time() - start_time
        print(f"[GPU] Training Time: {duration:.4f}s")
        
        # Calculate WSSSE (Within-Cluster Sum of Square Errors)
        final_distances = torch.cdist(X_tensor, centroids, p=2.0)
        min_distances = torch.min(final_distances, dim=1)[0]
        wssse = torch.sum(min_distances ** 2).item()
        
        results_str = f"PyTorch MPS GPU K-Means (k={k})\n"
        results_str += f"Number of samples: {n_samples}\n"
        results_str += f"Training Time: {duration:.4f}s\n"
        results_str += f"WSSSE: {wssse:.4f}\n\n"
        
        results_str += "Cluster Centers (Distance, Fare, Duration - Scaled):\n"
        centers_cpu = centroids.cpu().numpy()
        for center in centers_cpu:
            results_str += f"{center}\n"
            
        with open(os.path.join(results_dir, "clustering_gpu_pytorch_results.txt"), "w") as f:
            f.write(results_str)

        print("[GPU] PyTorch Clustering Complete.")
        
    except Exception as e:
        print(f"[GPU] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pytorch_kmeans()
