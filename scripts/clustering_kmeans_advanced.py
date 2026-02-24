import sys
import os
import time
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler

def run_advanced_kmeans():
    """
    Perform Advanced K-Means Clustering to identify 'Trip Personas'.
    Includes parameter exploration (k).
    """
    print("\n[ML] Starting Advanced K-Means Clustering...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "silver", "taxi_features.parquet")
    output_path = os.path.join(base_dir, "data", "gold", "clustered_trips_kmeans.parquet")
    results_dir = os.path.join(base_dir, "results")
    
    spark = SparkSession.builder \
        .appName("Clustering_Advanced_KMeans") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
        
    try:
        if not os.path.exists(input_path):
             print(f"Error: Silver data not found at {input_path}")
             sys.exit(1)
             
        df = spark.read.parquet(input_path)
        
        if os.environ.get("IS_TEST_RUN"):
             df = df.limit(50000)

        # Features for Clustering (Behavioral)
        features = ['trip_distance', 'fare_amount', 'trip_duration_min']
        
        assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
        
        # PIPELINE PREP
        df_assembled = assembler.transform(df)
        scaler_model = scaler.fit(df_assembled)
        df_scaled = scaler_model.transform(df_assembled).cache()
        
        # PARAMETER EXPLORATION
        k_values = [3, 4, 5, 6]
        best_k = -1
        best_silhouette = -1
        best_model = None
        
        evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette", distanceMeasure="squaredEuclidean")
        
        results_str = "K-Means Hyperparameter Exploration:\n"
        
        for k in k_values:
            print(f"Testing K-Means with k={k}...")
            start_time = time.time()
            kmeans = KMeans().setK(k).setSeed(42).setFeaturesCol("features").setMaxIter(20)
            model = kmeans.fit(df_scaled)
            predictions = model.transform(df_scaled)
            
            silhouette = evaluator.evaluate(predictions)
            duration = time.time() - start_time
            
            res = f"k={k} | Silhouette: {silhouette:.4f} | Time: {duration:.2f}s"
            print(res)
            results_str += res + "\n"
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
                best_model = model

        print(f"\nBest K: {best_k} with Silhouette: {best_silhouette:.4f}")
        results_str += f"\nBest K: {best_k} with Silhouette: {best_silhouette:.4f}\n"

        # Cluster Centers
        centers = best_model.clusterCenters()
        results_str += "\nBest Model Cluster Centers (Distance, Fare, Duration - Scaled):\n"
        for center in centers:
            results_str += str(center) + "\n"
            
        # Save Results
        with open(os.path.join(results_dir, "clustering_kmeans_advanced_results.txt"), "w") as f:
            f.write(results_str)

        # Save Best Predictions
        best_predictions = best_model.transform(df_scaled)
        final_df = best_predictions.select("prediction", *features)
        final_df.write.mode("overwrite").parquet(output_path)
        
        print("[ML] Advanced K-Means Complete.")
        
    except Exception as e:
        print(f"[ML] Error: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    run_advanced_kmeans()
