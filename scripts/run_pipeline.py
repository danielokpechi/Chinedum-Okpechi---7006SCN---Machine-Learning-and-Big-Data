import os
import subprocess
import sys

def run_pipeline():
    """
    Orchestrates the Dual-Stage Machine Learning Pipeline:
    1. Bronze (Ingest Raw -> Parquet)
    2. Silver (Clean/Feature Eng -> Parquet)
    3. Stage 1: Clustering (Discover Personas)
    4. Stage 2: Regression (Predict Fares)
    """
    print("="*60)
    print("STARTING DUAL-STAGE ML PIPELINE")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(base_dir, 'scripts')
    
    stages = [
        ("BRONZE JOINS", "ingest_zones.py"),
        ("BRONZE", "bronze_ingestion.py"),
        ("SILVER", "silver_processing.py"),
        ("CLUSTERING: ADVANCED K-MEANS", "clustering_kmeans_advanced.py"),
        ("CLUSTERING: BISECTING K-MEANS", "clustering_bisecting_kmeans.py"),
        ("CLUSTERING: GMM", "clustering_gmm.py"),
        ("CLUSTERING: GPU ACCELERATOR", "clustering_gpu_pytorch.py"),
        ("CLUSTERING: EVALUATION", "clustering_evaluation.py"),
        ("CLUSTERING: STABILITY", "clustering_stability_test.py"),
        ("GOLD (STAGE 2 REGRESSION TARGETS)", "gold_training_export.py"),
        ("STAGE 2: MODEL TRAINING", "model_training.py")
    ]
    
    wrapper_script = os.path.join(scripts_dir, "run_spark_with_conda.py")
    use_wrapper = os.path.exists(wrapper_script)
    
    for stage_name, script_name in stages:
        print(f"\n--- Running Stage: {stage_name} ({script_name}) ---")
        
        script_path = os.path.join(scripts_dir, script_name)
        if not os.path.exists(script_path):
             print(f"Error: Script {script_path} not found. Skipping.")
             continue
             
        if use_wrapper and script_name.endswith('.py') and script_name != "clustering_gpu_pytorch.py":
            # Use the wrapper to ensure correct PySpark/Conda env
            cmd = ["python", wrapper_script, script_name]
        else:
            cmd = ["python", script_path]
            
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"❌ Stage {stage_name} FAILED with return code {result.returncode}")
            sys.exit(result.returncode)
        else:
            print(f"✅ Stage {stage_name} SUCCESS")

    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
