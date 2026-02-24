import sys
import os
import time
from pyspark.sql import SparkSession
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler

def run_bisecting_kmeans():
    """
    Perform Bisecting K-Means Clustering.
    """
    print("\n[ML] Starting Bisecting K-Means Clustering...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "silver", "taxi_features.parquet")
    output_path = os.path.join(base_dir, "data", "gold", "clustered_trips_bisecting.parquet")
    results_dir = os.path.join(base_dir, "results")
    
    spark = SparkSession.builder \
        .appName("Clustering_Bisecting_KMeans") \
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
        
        # Bisecting K-Means
        k = 5
        print(f"Training Bisecting K-Means with k={k}...")
        start_time = time.time()
        
        bkm = BisectingKMeans().setK(k).setSeed(42).setFeaturesCol("features").setMaxIter(20)
        model = bkm.fit(df_scaled)
        predictions = model.transform(df_scaled)
        
        duration = time.time() - start_time
        
        evaluator = ClusteringEvaluator(featuresCol="features")
        silhouette = evaluator.evaluate(predictions)
        
        results_str = f"Bisecting K-Means (k={k})\n"
        results_str += f"Time: {duration:.2f}s\n"
        results_str += f"Silhouette Score: {silhouette:.4f}\n\n"
        
        centers = model.clusterCenters()
        results_str += "Cluster Centers (Distance, Fare, Duration - Scaled):\n"
        for center in centers:
            results_str += str(center) + "\n"
            
        with open(os.path.join(results_dir, "clustering_bisecting_kmeans_results.txt"), "w") as f:
            f.write(results_str)

        final_df = predictions.select("prediction", *features)
        final_df.write.mode("overwrite").parquet(output_path)
        
        print("[ML] Bisecting K-Means Complete.")
        
    except Exception as e:
        print(f"[ML] Error: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    run_bisecting_kmeans()
