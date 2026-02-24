
import os
import sys
import subprocess
import shutil

# 1. Detect and Configure Python Environment
print("--- Spark with Conda Wrapper ---")

# Try to detect Anaconda python automatically if current python is system python
# System python usually in /Library/... or /usr/bin/...
# Anaconda usually in /opt/anaconda3/... or ~/anaconda3/...

base_python = sys.executable
ANACONDA_PYTHON_CANDIDATES = [
    "/opt/anaconda3/bin/python3",
    os.path.expanduser("~/anaconda3/bin/python3"),
    os.path.expanduser("~/opt/anaconda3/bin/python3")
]

target_python = base_python
# Check if we are running in system python but numpy is missing (as seen in logs)
try:
    import numpy
except ImportError:
    # If numpy missing, try to find a better python
    print(f"NumPy missing in {base_python}. Searching for Anaconda...")
    for cand in ANACONDA_PYTHON_CANDIDATES:
        if os.path.exists(cand):
            print(f"Found Anaconda Python at: {cand}")
            target_python = cand
            break

# If we changed python, we should ideally re-spawn this script with the new python
# but simpler is just to tell Spark to use THAT python for workers/driver
# AND strictly use its site-packages? 
# Spark driver runs in THIS process if we use 'pyspark' library imports directly?
# No, `SparkSession` in local mode runs in the JVM but python workers are spawned.
# The DRIVER (this script) needs numpy if we use numpy here.

# CRITICAL: If this script itself cannot import numpy, we can't verify dependencies or run logic that uses numpy.
# We must RE-EXECUTE this script with the correct python if we are in the wrong one.

if target_python != base_python:
    print(f"Re-launching script with {target_python}...")
    # Pass all args
    args = [target_python] + sys.argv
    os.execv(target_python, args)

CURRENT_PYTHON = sys.executable
print(f"Current Python Interpreter: {CURRENT_PYTHON}")

# Set environment variables for PySpark to use the same Python
os.environ['PYSPARK_PYTHON'] = CURRENT_PYTHON
os.environ['PYSPARK_DRIVER_PYTHON'] = CURRENT_PYTHON

# Ensure Java 17 is used
JAVA_HOME_PATH = "/opt/homebrew/opt/openjdk@17"
if os.path.exists(JAVA_HOME_PATH):
    print(f"Setting JAVA_HOME to {JAVA_HOME_PATH}")
    os.environ['JAVA_HOME'] = JAVA_HOME_PATH
    # Also update PATH to prepend Java 17
    os.environ['PATH'] = f"{JAVA_HOME_PATH}/bin:{os.environ['PATH']}"
else:
    print(f"Warning: JAVA_HOME {JAVA_HOME_PATH} not found. Relying on system default.")


# 2. Verify Dependencies
required_packages = ['numpy', 'pyarrow']
print("\nVerifying dependencies...")
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"  [OK] {pkg}")
    except ImportError:
        print(f"  [MISSING] {pkg}")
        print(f"Error: Required package '{pkg}' is missing in the current environment.")
        print(f"Please install it using: pip install {pkg}")
        sys.exit(1)
        
# Check numpy version just in case
import numpy
print(f"  NumPy Version: {numpy.__version__}")

# 3. Initialize Spark Session
try:
    from pyspark.sql import SparkSession
except ImportError:
    print("Error: PySpark not found. Please install it: pip install pyspark")
    sys.exit(1)

print("\nInitializing Spark Session...")
try:
    spark = SparkSession.builder \
        .appName("Spark_Conda_Wrapper") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
        .getOrCreate()
        
    print(f"  Spark Version: {spark.version}")
    print(f"  Java Version: {spark.conf.get('spark.executor.extraJavaOptions', 'Not Set/Default')}") # Hard to get exact Java version from SparkConf easily without JVM access
    print("  Spark Session Active.")
    
except Exception as e:
    print(f"Failed to initialize Spark: {e}")
    sys.exit(1)

# 4. Execute MLlib Logic
# 4. Execute MLlib Logic
if len(sys.argv) > 1:
    script_name = sys.argv[1]
else:
    # Default to training if no arg provided
    script_name = "mllib_training_and_eval.py"
    
TARGET_SCRIPT = os.path.join(os.path.dirname(__file__), script_name)
print(f"\nExecuting: {TARGET_SCRIPT}")
print("="*60)

try:
    # Read and exec the script in the current process
    # This allows it to use the SparkSession we just created (since getOrCreate will return it)
    # and ensures it runs in the same Python environment.
    with open(TARGET_SCRIPT, 'r') as f:
        script_content = f.read()
        
    # We execute in a global dictionary to avoid pollution, but pass 'spark' if needed? 
    # Actually, mllib calls getOrCreate(), so it will pick up 'spark' from the internal cache.
    # We pass globals() to allow imports to work standardly.
    exec(script_content, globals())
    
except Exception as e:
    print(f"\nExecution failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\n" + "="*60)
    print("Stopping Spark Session...")
    spark.stop()
    print("Done.")
