import sys
import os
import time
from pyspark.sql import SparkSession
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler

def run_gmm():
    """
    Perform Gaussian Mixture Model (GMM) Clustering.
    """
    print("\n[ML] Starting GMM Clustering...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "silver", "taxi_features.parquet")
    output_path = os.path.join(base_dir, "data", "gold", "clustered_trips_gmm.parquet")
    results_dir = os.path.join(base_dir, "results")
    
    spark = SparkSession.builder \
        .appName("Clustering_GMM") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
        
    try:
        if not os.path.exists(input_path):
             print(f"Error: Silver data not found at {input_path}")
             sys.exit(1)
             
        df = spark.read.parquet(input_path)
        
        if os.environ.get("IS_TEST_RUN"):
             df = df.limit(50000)

        # Features for Clustering
        features = ['trip_distance', 'fare_amount', 'trip_duration_min']
        
        assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
        
        df_assembled = assembler.transform(df)
        scaler_model = scaler.fit(df_assembled)
        df_scaled = scaler_model.transform(df_assembled).cache()
        
        # GMM
        k = 5
        print(f"Training Gaussian Mixture Model with k={k}...")
        start_time = time.time()
        
        gmm = GaussianMixture().setK(k).setSeed(42).setFeaturesCol("features").setMaxIter(20)
        model = gmm.fit(df_scaled)
        predictions = model.transform(df_scaled)
        
        duration = time.time() - start_time
        
        evaluator = ClusteringEvaluator(featuresCol="features")
        silhouette = evaluator.evaluate(predictions)
        
        results_str = f"Gaussian Mixture Model (k={k})\n"
        results_str += f"Time: {duration:.2f}s\n"
        results_str += f"Silhouette Score: {silhouette:.4f}\n\n"
        
        results_str += "Model Weights (Priors):\n"
        for i, weight in enumerate(model.weights):
            results_str += f"Cluster {i}: {weight:.4f}\n"

        results_str += "\nGaussians (Means and Covariances):\n"
        for i, (avg, cov) in enumerate(zip(model.gaussiansDF.select("mean").collect(), model.gaussiansDF.select("cov").collect())):
            results_str += f"Cluster {i} Mean: {avg[0]}\n"
            
        with open(os.path.join(results_dir, "clustering_gmm_results.txt"), "w") as f:
            f.write(results_str)

        final_df = predictions.select("prediction", *features)
        final_df.write.mode("overwrite").parquet(output_path)
        
        print("[ML] GMM Clustering Complete.")
        
    except Exception as e:
        print(f"[ML] Error: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    run_gmm()
