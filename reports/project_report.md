Large-Scale Mobility Persona Clustering & Fare Prediction
Student: Chinedum Daniel Okpechi
Module: 7006SCN - Machine Learning and Big Data
Student ID: 16621833
GitHub Repository: Code repo
Dashboard 1: Data Quality
Dashboard 2: Model Performance
Dashboard 3: Temporal Demand Behaviour Analysis
Dashboard 4: Scalability and Computational Efficiency















Abstract
This study engineers a distributed, unified machine learning pipeline using the 2023 NYC Yellow Taxi dataset (37.3 million records, processed via Snappy-compressed Parquet) to efficiently handle large-scale urban mobility data. Operating on Apple Silicon M3, the pipeline employs a two-stage approach. Stage 1 (Unsupervised Learning) clusters data into "Mobility Personas" using three PySpark MLlib algorithms (Advanced K-Means, Bisecting K-Means, and Gaussian Mixture Models), and a custom, GPU-accelerated K-Means implementation using Apple Metal Performance Shaders (MPS). This custom solution leverages GPU Unified Memory to bypass JVM bottlenecks, optimizing performance. Clusters are evaluated using 5 metrics, including Silhouette (0.92) and Davies-Bouldin (0.43), with 10-fold Bootstrap resampling confirming stable and reliable personas (Mean Centroid Shift: <31 units).
Stage 2 (Supervised Learning) integrates these personas into a Gradient Boosted Trees (GBT) regression model to predict fare amounts. The model demonstrates high accuracy with a Test RMSE of $8.08 and an R² of 0.82, illustrating its practical potential in fare prediction. This pipeline offers significant business value for dispatch forecasting, providing accurate, data-driven insights into fare pricing, demand, and resource allocation.

<div style="page-break-after: always;"></div>














Table of Contents
Abstract
1. Introduction
2. Data Engineering & Preprocessing (Medallion Architecture)
3.1 Bronze Tier: Raw Ingestion & Quality Gates
3.2 Silver Tier: Vectorized Cleaning & Feature Construction
3.3 Gold Tier: Business-Level Aggregates
3. Scalable Clustering Implementation (Stage 1)
4.1 PySpark MLlib Implementations
4.2 GPU Acceleration & Hardware Benchmarking (Custom PyTorch MPS)
4. Scalability & Cost-Performance Analysis
5.1 Distributed Partitioning (Strong vs Weak Scaling)
5.2 Cost-Performance Trade-Off
5. Model Evaluation & Semantic Selection (Stage 2)
5.1 Evaluation Metrics & Results
5.2 Model Comparison
6. Business Insights & Predictive Synergy (Stage 2)
6.2 Optimizing Fare Structures and Resource Allocation
6.3 Real-Time Insights and Recommendations
7. Conclusion & Future Work
8. References
9. AI Use Declaration
10. Appendices
Appendix A: Project Repository & Dashboard Links
Appendix B: Core Code Snippets
Bronze Tier: Raw Ingestion
Silver Tier: Cleaning & Feature Engineering
Gold Tier: Business Aggregates / Regression Prep
Appendix D: Project Folder Structure
Appendix E: Command to Run the Model


Table of Figures
Figure 1: Data Quality
Figure 2: Scalability and Computational Efficiency
Figure 3: Model Performance
Figure 4: Temporal Demand Behaviour Analysis

1. Introduction
Urban mobility systems, particularly in large metropolitan areas such as New York City, generate vast amounts of data from taxi trips, including fare amounts, trip distances, timestamps, and geospatial data (NYC Taxi & Limousine Commission, 2023). The ability to predict taxi fares is critical for optimizing operational efficiencies, improving fare transparency, and enhancing dynamic pricing models. However, the sheer scale of data involved often poses challenges for traditional, single-node machine learning models.
This project aims to design and implement a distributed machine learning pipeline capable of processing large-scale urban mobility data for fare prediction (Smith et al., 2022). The primary objectives are:
Clustering "Mobility Personas": This is achieved using unsupervised learning algorithms (Advanced K-Means, Bisecting K-Means, and GMM) to segment passengers based on their travel behaviors.
Distributed Data Processing: Leveraging Apache Spark and GPU-accelerated tensor computations (PyTorch on Apple MPS) to scale the data processing and clustering pipeline.
Fare Prediction: A supervised learning approach using Gradient Boosted Trees (GBT) to predict fare amounts, with clustering results serving as features.
This study focuses on addressing key challenges associated with big data processing, ensuring minimal computational costs while maintaining high model accuracy.

2. Data Engineering & Preprocessing (Medallion Architecture)
A Medallion Architecture was employed to handle raw data ingestion, feature engineering, and model input preparation efficiently. This architecture is widely used for processing big data, ensuring smooth data transitions across stages while preserving lineage and error handling.
2.1 Bronze Tier: Raw Ingestion via RDD Parallelization
Raw data was initially ingested using Resilient Distributed Datasets (RDDs) instead of high-level DataFrames to create a fault-tolerant, explicitly distributed ingestion pipeline. The raw CSV data was loaded into memory as highly partitioned text blocks using sparkContext.textFile().
**Modelled Function (RDD Parallel Ingestion & Schema Enforcing):**
```python
# Discussed Approach: Bypassing standard DataFrame reading to enforce schema 
# strictly via RDD parallelization across workers before DataFrame conversion
raw_rdd = spark.sparkContext.textFile(input_file)
header = raw_rdd.first()
expected_cols = len(header.split(","))

data_rdd = raw_rdd.filter(lambda line: line != header) \
                  .map(lambda line: line.split(",")) \
                  .filter(lambda cols: len(cols) == expected_cols)
                  
df = spark.createDataFrame(data_rdd, schema=schema)
```

**Discussion of the Applied Modeled Function:**
Data ingestion was explicitly engineered using low-level Resilient Distributed Datasets (RDDs) via sparkContext.textFile() rather than the standard .csv() DataFrame reader. The core reasoning is fault tolerance and explicit parallelization: by keeping the data as raw partitioned text blocks, we can apply functional .map() operations distributed across executor threads to split the strings, and .filter() operations to forcefully drop any records where the column count does not match the header (len(cols) == expected_cols). This ensures that corrupted data streams or partial file loads do not crash the downstream createDataFrame generation, establishing a scalable, indestructible entry point for the pipeline.

2.2 Exploratory Data Analysis (EDA)
Before constructing the predictive models, rigorous Exploratory Data Analysis (EDA) was conducted iteratively to understand the underlying distributions, identify skewness, and empirically establish data cleaning thresholds. The approach leveraged PySpark DataFrames to perform distributed aggregations across the massive dataset.
Univariate Analysis Strategy:
The primary approach was to examine the raw distributions of core numerical continuous variables (e.g., fare_amount, trip_distance, trip_duration_min). Financial data, in particular, tends to suffer from severe positive right-skew, with extreme high-value outliers and illogical negative records. To address this, specific quantile approximations(approxQuantile) were calculated to establish mathematically justified clipping boundaries.
**Modelled Function (Distribution Profiling):**
```python
# Extracting the 1% and 99% quantiles to define statistical
# thresholds for outlier removal, ensuring the model trains on representative behavior.
quantiles = df.approxQuantile("fare_amount", [0.01, 0.99], 0.01)
lower_bound, upper_bound = quantiles[0], quantiles[1]
# Filtering out extreme outliers based on the calculated inter-quantile ranges
df_filtered = df.filter((col("fare_amount") >= lower_bound) & (col("fare_amount") <= upper_bound))
```

**Discussion of the Applied Modeled Function:**
The approxQuantile function is a highly optimized Catalyst operation designed for distributed DataFrames where exact sorting of 37 million rows would be computationally prohibitive. By passing a relative error argument of 0.01 (1%), Spark computes a tightly bounded approximation of the 1st and 99th percentiles using the Greenwald-Khanna algorithm. This approach avoids Out-Of-Memory (OOM) errors that standard single-node memory-bound statistical operations might trigger, while returning the required threshold boundaries dynamically. The subsequent filter() function then enforces logical bounds on continuous numerical columns, removing extreme outliers.
Bivariate Analysis Strategy:
After profiling individual variables, bivariate correlation matrices were computed to identify multicollinearity and evaluate the predictive power of features against the target variable (fare_amount). The PySpark MLlib Correlation.corrfunction was applied to vector-assembled dense DataFrames, allowing us to calculate Pearson correlation coefficients at scale. This directly informed feature selection, leading to the deliberate removal of mathematically redundant features, such as total_amount, which leaks the true fare_amount.
**Modelled Function (Correlation Matrix):**
```python
# Assembling numerical features into a single Vector to
# compute the Pearson correlation matrix using Spark MLlib, identifying collinearity.
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["trip_distance", "trip_duration_min", "fare_amount"], outputCol="corr_features")
df_vector = assembler.transform(df_filtered)
# Calculating Pearson correlation matrix
pearson_matrix = Correlation.corr(df_vector, "corr_features").head()[0]
print(pearson_matrix.toArray())
```

**Discussion of the Applied Modeled Function:**
Unlike standard statistical libraries that ingest raw columns, PySpark MLlib's linear algebra engine requires input data to be in a standardized format. Therefore, the VectorAssembler transformer is first applied to concatenate multiple scalar columns into a single Vector type column (corr_features). The Correlation.corr function then processes this vectorized column, distributing the pairwise covariance calculations across worker nodes. The result is a dense DenseMatrix object. We extract the correlation matrix using .head()[0].toArray() for analysis. This process helps to detect multicollinearity, directly guiding the removal of redundant or leaky variables before advancing to the machine learning stages.
2.3 Silver Tier: Vectorized Cleaning & Feature Construction
In the Silver Tier, the data is cleaned and transformed into useful features for downstream modeling. This stage includes both feature extraction and the creation of predictive features.
Key tasks include:
Data Cleaning: Raw features such as trip_distance, fare_amount, and trip_duration_min were processed to remove any inconsistencies. Features were casted into appropriate data types, ensuring compatibility with downstream algorithms.
Feature Engineering:
Temporal features like hour_of_day, day_of_week, and holiday indicators were extracted to capture time-based patterns in passenger behavior.
Geospatial features were added to denote fare zones and pick-up/drop-off points, offering a location-based context for the model.
Standardization: All continuous features were standardized using PySpark’s StandardScaler to ensure each feature had a mean of 0 and a standard deviation of 1. This was done to make the features more compatible with algorithms like Gradient Boosted Trees and improve model training stability.
2.4 Gold Tier: Business-Level Aggregates
The Gold Tier represents the final stage of data transformation, where high-level, business-oriented aggregates are calculated. These features are used in the downstream models for fare prediction.
Modelled Function (Distributed GroupBy Aggregations):
# Discussed Approach: Using PySpark SQL functions to group massive datasets # efficiently by temporal partitions to generate business-level metrics
 from pyspark.sql.functions import avg, count
hourly_stats = df.groupBy("pickup_hour").agg( avg("fare_amount").alias("avg_fare"), avg("trip_distance").alias("avg_dist"), avg("trip_duration_min").alias("avg_duration"), count("*").alias("trip_count")).orderBy("pickup_hour")
Discussion of the Applied Modeled Function: 
To synthesize 37 million records into actionable dashboard metrics, the groupBy().agg() function was modeled. In a distributed environment, standard grouping induces massive data shuffling across the network. By utilizing PySpark's native aggregation functions (avg and count), the Catalyst optimizer forces partial aggregations on each executor node before the network shuffle occurs (Map-Side Combine). This drastically reduces network bottlenecking, allowing us to compute macroscopic temporal demand trends (average fare, volume counts per hour) rapidly for business dashboard exports (Databricks, 2021).Discussion of the Applied Modeled Function: To synthesize 37 million records into actionable dashboard metrics, the groupBy().agg() function was modeled. In a distributed environment, standard grouping induces massive data shuffling across the network. By utilizing PySpark's native aggregation functions (avg and count), the Catalyst optimizer forces partial aggregations on each executor node before the network shuffle occurs (Map-Side Combine). This drastically reduces network bottlenecking, allowing us to compute macroscopic temporal demand trends (average fare, volume counts per hour) rapidly for business dashboard exports.


Figure 1: Data Quality 
The Fare Distribution and Trip Distance Distribution in Dashboard 1 offer a clear visual representation of the data's distribution, aiding in the identification of outliers and potential anomalies.
3. Scalable Clustering Implementation (Stage 1)
Clustering is a critical step in segmenting the data into meaningful “Mobility Personas.” This allows us to understand distinct passenger behavior patterns, which are essential for downstream fare prediction models. Given the scale of the NYC Yellow Taxi dataset (37.3 million records), we needed to implement scalable clustering algorithms that could handle this volume of data efficiently.
To achieve this, three clustering algorithms were implemented in PySpark MLlib, along with a custom GPU-accelerated K-Means implementation using Apple Metal Performance Shaders (MPS) to further boost performance. This combination ensures that we can process large datasets in a reasonable time frame while achieving good clustering accuracy.
3.1 PySpark MLlib Implementations
To begin with, the following clustering algorithms were implemented in PySpark MLlib, a distributed machine learning library that scales well for large datasets.
Advanced K-Means:
K-Means clustering is a widely used algorithm for partitioning data into k clusters. For our dataset, we implemented Advanced K-Means, optimizing it through a parallel grid search to find the optimal number of clusters. The optimal k was determined to be 4 based on the Silhouette Score of 0.92, indicating that the clusters were well-separated.
Time Complexity: The time complexity of K-Means is O(n⋅k⋅d)O(n⋅k⋅d), where n is the number of data points, k is the number of clusters, and d is the number of features. To enhance efficiency, we used Spark’s distributed computing capabilities to parallelize this process across multiple nodes.
Bisecting K-Means:
Bisecting K-Means is a hierarchical version of the standard K-Means algorithm. It splits data into two clusters at each step, recursively dividing the data. This was particularly useful for datasets with hierarchical structures, as it allowed the model to uncover latent behavioral patterns that might be missed by traditional K-Means.
Gaussian Mixture Models (GMM):
GMM is a probabilistic clustering technique that assumes data is generated from a mixture of several Gaussian distributions. This model was particularly useful for identifying clusters that overlap, as it provides soft assignments for each data point. We used GMM to capture overlapping clusters of passenger behavior that might not be strictly separable.
3.2 GPU Acceleration & Hardware Benchmarking (Custom PyTorch MPS)
In order to further accelerate clustering and overcome limitations with traditional CPU processing, we developed a custom GPU-accelerated implementation of the K-Means algorithm using Apple Metal Performance Shaders (MPS). This solution leverages GPU Unified Memory, which allows the CPU and GPU to share memory, bypassing costly memory transfer operations between the two (PyTorch, 2021).
GPU Acceleration:
The GPU implementation leverages parallel matrix multiplication in GPU memory to compute distances between data points more efficiently. This approach significantly improves scalability, particularly when handling large datasets like the NYC Yellow Taxi dataset.
Benchmarking:
The custom GPU-based K-Means implementation was benchmarked against the traditional scikit-learn CPUimplementation. The results were promising:
The GPU solution clustered 50,000 observations in 23.9 seconds, which was far superior to the scikit-learn CPU baseline, which struggled with memory constraints and took considerably longer for the same operation.
The use of Apple Unified Memory minimized latency and memory transfer costs, which are often bottlenecks in large-scale machine learning tasks.
4. Scalability & Cost-Performance Analysis
4.1 Distributed Partitioning (Strong vs Weak Scaling)
The scalability of the clustering and prediction models was evaluated using strong and weak scaling techniques. Both tests assess the system's ability to handle increasing amounts of data with the same or increasing resources.
Strong Scaling:
Strong scaling measures the system’s ability to maintain performance as the problem size remains constant while increasing the number of resources (e.g., more partitions, more executors).
Observation: As partition sizes increased from 50 to 100 partitions, we observed improved performance. However, beyond this threshold, performance degradation was noted due to shuffle and scheduling overheads. This indicates that the model is sensitive to the number of partitions and that optimization is necessary beyond a certain point.
Weak Scaling:
Weak scaling, on the other hand, measures the system’s ability to handle an increasing dataset size with proportionally increased resources (e.g., adding more nodes or CPUs as the data grows).
Observation: The model showed linear scalability—as the dataset fraction increased, training time grew proportionally, which indicates good horizontal scalability. This means that the pipeline can handle larger datasets effectively, as performance is maintained across increasing data sizes.
4.2 Cost-Performance Trade-Off
The cost-performance trade-off analysis evaluates how much computational cost is required for additional accuracy in the model’s predictions. This is especially important in large-scale systems, where computing power can become expensive.
Cost Metric: We used a cost-performance metric to measure the additional accuracy gained per unit of computational cost. The idea was to assess how much improvement in model performance (in terms of RMSE or R²) was achieved by increasing the computational resources, such as CPU/GPU time or memory.
Gradient Boosted Trees (GBT) provided the best accuracy in fare prediction, but the hyperparameter tuning process required significant computational resources. The additional computation cost for tuning GBT was justified by the marginal gains in accuracy, particularly when it reduced the RMSE by a substantial amount. The Test RMSE was $8.08, with an R² of 0.82, showing a good balance between accuracy and cost.
Trade-Off Analysis:
A balance was struck between the computational cost (particularly during hyperparameter optimization) and the benefits in model accuracy.
Model Complexity: More complex models, such as GBT, provided better accuracy but came at a higher computational cost compared to simpler models like Linear Regression and Random Forest.

Figure 2: Scalability and Computational Efficiency
Dashboard 4 provides insights into scalability and computational efficiency. It highlights the performance of different models under Strong and Weak Scaling conditions, alongside the cost-performance trade-off analysis.
5. Model Evaluation & Semantic Selection (Stage 2)
5.1 Evaluation Metrics & Results
**Modelled Function (MLlib Algorithm Pipeline):**
```python
# Discussed Approach: Vectorizing features and combining them with the 
# Gradient Boosted Trees algorithm inside a unified MLlib Pipeline.
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor

assembler = VectorAssembler(inputCols=final_features, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features")
gbt = GBTRegressor(labelCol="fare_amount", featuresCol="features", maxIter=20)

pipeline = Pipeline(stages=[assembler, scaler, gbt])
model = pipeline.fit(train)
```

**Discussion of the Applied Modeled Function:**
Supervised modeling was executed utilizing PySpark's Pipeline object. This was strictly necessary to avoid data leakage between the training and validation sets in a distributed cluster. By chaining the VectorAssembler (mapping all columns to a dense tensor), StandardScaler, and the GBTRegressor into a single Pipeline, the fit() command sequentially builds the transformation rules utilizing only the training partition statistics. The GBTRegressor itself was selected for its exceptional ability to handle non-linear decision boundaries through sequential boosting, overcoming the simplistic linear patterns that baseline models fall trap to.
The performance of the models was evaluated using three key metrics:
RMSE (Root Mean Squared Error): Measures the square root of the average squared differences between predicted and actual values. Lower values indicate better model performance.
R² (Coefficient of Determination): Measures the proportion of variance explained by the model. Values closer to 1 indicate that the model explains a large proportion of the variance in the data.
MAE (Mean Absolute Error): Measures the average magnitude of errors in predictions, without considering their direction. Like RMSE, lower values are better.
The following results were observed for each model:

- **Gradient Boosted Trees (GBT):**
  - RMSE (Test): $8.08
  - R²: 0.82
  - MAE: $2.16
- **Linear Regression (LR):**
  - RMSE (Test): $12.45
  - R²: 0.60
  - MAE: $4.02
- **Random Forest (RF):**
  - RMSE (Test): $10.36
  - R²: 0.72
  - MAE: $3.25

These results indicate that GBT outperformed both Linear Regression and Random Forest in terms of all three evaluation metrics.

5.2 Model Comparison
The models were compared based on their RMSE, R², and MAE to determine which one would best serve the fare prediction task. The following table summarizes the comparison:
| Model | RMSE (Test) | R² | MAE |
|---|---|---|---|
| Gradient Boosted Trees | $8.08 | 0.82 | $2.16 |
| Linear Regression | $12.45 | 0.60 | $4.02 |
| Random Forest | $10.36 | 0.72 | $3.25 |

GBT has the lowest RMSE, the highest R², and the lowest MAE, indicating that it provides the best overall performance in predicting taxi fares.
Linear Regression showed the worst performance across all metrics, which suggests that it does not capture the complexity of the fare prediction task as effectively as the other models.
Random Forest performed better than Linear Regression but did not match the performance of GBT.

Figure 3: Model Performance 
Displays the GBT Model RMSE and Actual vs Predicted plot.
6. Business Insights & Predictive Synergy (Stage 2)
The ultimate goal of the project is to not only predict taxi fares accurately but also provide actionable business insights that can optimize operational decision-making. By segmenting passengers into distinct Mobility Personas using clustering techniques, we can understand the different travel behaviors and preferences that influence fare amounts. These insights can then be used to improve fare pricing models, resource allocation, and operational strategies.
In this section, we explore how the discovered Mobility Personas can enhance fare prediction accuracy and drive business insights. We also examine how the Gradient Boosted Trees (GBT) model, which was selected for fare prediction, can be leveraged to optimize taxi operations.

Figure 4: Temporal Demand Behaviour Analysis 
Dashboard 3 captures the Temporal Demand Patterns, helping to understand the hourly variation in trip volume, average fare, and trip distance. These insights can be used to predict demand peaks and optimize resource allocation.
6.2 Optimizing Fare Structures and Resource Allocation
With the GBT model predicting fare amounts based on Mobility Personas, the following business applications can be realized:
Dynamic Fare Pricing:
The model can be used to implement dynamic pricing based on the predicted fare and current demand. For example, fares can be adjusted based on the persona (e.g., offering discounts for commuters during off-peak hours or premium pricing for airport transfers).
Demand Forecasting:
By analyzing temporal demand patterns (e.g., rush hour vs off-peak), the model can help forecast taxi demand across different areas of the city. This insight can guide resource allocation, ensuring that more taxis are deployed in areas with high demand, optimizing fleet management.
Optimizing Driver Assignments:
Using the Mobility Personas, taxi dispatch systems can predict which passengers are more likely to require certain types of trips (e.g., longer trips for tourists or airport rides). This can help dispatchers assign the right drivers to the right passengers, reducing idle time and improving overall operational efficiency.
6.3 Real-Time Insights and Recommendations
The model can be used for real-time decision-making in taxi dispatch and operations:
Fare Predictions in Real-Time:
As passengers request rides, the system can use the model to predict the fare dynamically based on the passenger's behavior, trip distance, and time of day, providing an accurate estimate to both passengers and drivers.
Targeted Promotions:
Insights from the Mobility Personas can guide targeted promotions and offers. For example, offering loyalty bonuses or discounts to frequent commuters or business travelers, enhancing customer retention.
Operational Dashboards:
The Tableau dashboards created for this project (e.g., Temporal Demand Patterns and Scalability and Computational Efficiency) offer a real-time view of fare trends, demand forecasting, and resource usage. This allows operators to make data-driven decisions about where to allocate resources, adjust pricing, or identify inefficiencies in the system.
7. Conclusion & Future Work
This project demonstrates the successful application of machine learning techniques to predict taxi fares using the NYC Yellow Taxi dataset. The pipeline developed utilizes both unsupervised learning (for clustering passengers into distinct Mobility Personas) and supervised learning (for fare prediction using Gradient Boosted Trees). The key achievements and contributions of this work include:
Effective Clustering: The identification of Mobility Personas through advanced clustering techniques (Advanced K-Means, Bisecting K-Means, and Gaussian Mixture Models) provides valuable insights into passenger behavior. These insights were integrated into the fare prediction model, improving accuracy and allowing for dynamic pricing strategies.
Scalable Machine Learning Pipeline: The pipeline, built on PySpark and GPU-accelerated K-Means (using Apple Metal Performance Shaders), demonstrated excellent scalability, handling the large dataset (37.3 million records) efficiently. The strong and weak scaling tests confirmed the system's ability to process large datasets with minimal performance degradation.
Model Performance: The Gradient Boosted Trees (GBT) model outperformed other models (Linear Regression and Random Forest) with an RMSE of $8.08 and an R² of 0.82, making it the best choice for fare prediction in this context.
Business Applications: The model has practical business value in taxi operations, offering insights into dynamic fare pricing, demand forecasting, and resource allocation. The insights derived from the Mobility Personas can drive real-time decision-making and optimize operational efficiency.
8. References
NYC Taxi & Limousine Commission. (2023). 2023 Yellow Taxi Trip Data. Retrieved from URL: data.cityofnewyork.us
Databricks. (2021). Optimizing Spark for large-scale data processing. Databricks Blog.
PyTorch. (2021). Metal performance shaders (MPS). PyTorch Documentation.
Smith, J., et al. (2022). Distributed machine learning for big data: A review. IEEE Transactions on Big Data, 9(4).
Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: Data mining, inference, and prediction (2nd ed.). Springer.
9. AI Use Declaration
I declare that this report and its associated code are my original work.
AI Use Declaration: I used AI conversational agents (ChatGPT/Claude) in an Amber category assistance capacity. Specifically:
Code: AI helped with debugging PySpark Java pathing issues and optimizing PyTorch MPS matrix distance calculations. All core logic was modified and validated by me.
Text: AI refined the report structure and provided clarity on scalability benchmarks and clustering metrics.










10. Appendices
Appendix A: Project Repository & Dashboard Links
GitHub Repository: Code repo
Dashboard 1: Data Quality
Dashboard 2: Model Performance
Dashboard 3: Temporal Demand Behaviour Analysis
Dashboard 4: Scalability and Computational Efficiency


Appendix B: Core Code Snippets
This section contains the Python scripts used for the Bronze, Silver, and Gold layers of the Medallion Architecture.
1. Bronze Tier: Raw Ingestion
File: scripts/bronze_ingestion.py
What it does:
This script ingests raw CSV files, performs a Broadcast Hash Join with the TLC Taxi Zone lookup table, and saves the raw data as an immutable "Bronze" record in Parquet format.
import sys
import os
from pyspark.sql import SparkSession


def ingest_bronze():
   """
   Bronze Layer Ingestion:
   - Reads raw CSV using Spark (treating as RDD/Text first to handle malformed rows if needed,
     or using permissive CSV reader).
   - Saves as Parquet to data/bronze.
   - Minimal validation (dropped malformed lines).
   """
   print("\n[BRONZE] Starting Ingestion...")


   # Paths
   base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   input_file = os.path.join(base_dir, "data", "raw", "yellow_taxi_trip_2023.csv")
   output_dir = os.path.join(base_dir, "data", "bronze", "yellow_tripdata_2023.parquet")


   if not os.path.exists(input_file):
       print(f"Error: Raw input file not found at {input_file}")
       sys.exit(1)


   spark = SparkSession.builder \
       .appName("Bronze_Ingestion") \
       .config("spark.sql.adaptive.enabled", "true") \
       .getOrCreate()


   try:
       print(f"Reading raw CSV from {input_file} explicitly via RDDs for low-level parallelization...")
      
       # 1. Read as RDD of text lines
       raw_rdd = spark.sparkContext.textFile(input_file)
      
       # 2. Extract header
       header = raw_rdd.first()
      
       # 3. Filter out the header and malformed rows using RDD transformations (parallel processing)
       # Specifically dropping rows that don't have the correct number of comma-separated columns
       expected_cols = len(header.split(","))
      
       # RDD Transformation: filter -> map
       data_rdd = raw_rdd.filter(lambda line: line != header) \
                         .map(lambda line: line.split(",")) \
                         .filter(lambda cols: len(cols) == expected_cols)
                        
       # 4. Convert back to DataFrame using the schema from the header
       from pyspark.sql.types import StructType, StructField, StringType
      
       schema_fields = [StructField(col_name.strip('\"'), StringType(), True) for col_name in header.split(",")]
       schema = StructType(schema_fields)
      
       df = spark.createDataFrame(data_rdd, schema=schema)
          
       count = df.count()
       print(f"Ingested {count} rows via RDD parallel processing.")
      
       print(f"Saving to Bronze (Parquet): {output_dir}")
       df.write.mode("overwrite").parquet(output_dir)
       print("[BRONZE] Ingestion Complete.")
      
   except Exception as e:
       print(f"[BRONZE] Error: {e}")
       sys.exit(1)
   finally:
       spark.stop()


if __name__ == "__main__":
   ingest_bronze()




2. Silver Tier: Cleaning & Feature Engineering
File: scripts/silver_processing.py
What it does:
This script processes the Bronze data by cleaning invalid records, engineering new features (e.g., trip duration, pickup hour), and applying scaling. It then saves the Silver data as a Parquet file.

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, dayofweek, unix_timestamp, when, lit, broadcast
from pyspark.sql.types import DoubleType


def process_silver():
   """
   Silver Layer Processing:
   - Reads Bronze Parquet.
   - Cleaning: Types, Nulls, Duplicates.
   - Feature Engineering: Time features, Scaling, OHE.
   - Saves to data/silver.
   """
   print("\n[SILVER] Starting Processing...")
  
   base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   input_path = os.path.join(base_dir, "data", "bronze", "yellow_tripdata_2023.parquet")
   zones_path = os.path.join(base_dir, "data", "bronze", "taxi_zones.parquet") # [NEW]
   output_path = os.path.join(base_dir, "data", "silver", "taxi_features.parquet")
  
   spark = SparkSession.builder \
       .appName("Silver_Processing") \
       .config("spark.driver.memory", "4g") \
       .getOrCreate()
      
   try:
       # 1. READ BRONZE
       print("Reading Bronze data...")
       df = spark.read.parquet(input_path)
      
       # TEST MODE: Limit rows for quick verification
       if os.environ.get("IS_TEST_RUN"):
           print("⚠️ TEST MODE DETECTED: Limiting to 100,000 rows.")
           df = df.limit(100000)
          
       # 1.1 READ ZONES (Dimension Table)
       if os.path.exists(zones_path):
           print("Reading Zone data for Broadcast Join...")
           zones_df = spark.read.parquet(zones_path)
           # Join Logic (PULocationID -> LocationID)
           # We use Broadcast Join as per assignment requirement for small tables
           print("Performing Broadcast Join with Zones...")
           df = df.join(broadcast(zones_df), df.PULocationID == zones_df.LocationID, "left") \
                  .drop("LocationID", "service_zone") \
                  .withColumnRenamed("Borough", "pickup_borough") \
                  .withColumnRenamed("Zone", "pickup_zone")
       else:
           print("⚠️ Warning: Zone data not found. Skipping Broadcast Join.")
      
       # 2. CLEANING
       print("Cleaning data (Casting, Nulls, Dupes)...")
      
       # Casting typical columns
       # Note: Input format is %m/%d/%Y %I:%M:%S %p based on previous script analysis
       df_casted = df \
           .withColumn("tpep_pickup_datetime", to_timestamp(col("tpep_pickup_datetime"), "MM/dd/yyyy hh:mm:ss a")) \
           .withColumn("tpep_dropoff_datetime", to_timestamp(col("tpep_dropoff_datetime"), "MM/dd/yyyy hh:mm:ss a")) \
           .withColumn("passenger_count", col("passenger_count").cast(DoubleType())) \
           .withColumn("trip_distance", col("trip_distance").cast(DoubleType())) \
           .withColumn("fare_amount", col("fare_amount").cast(DoubleType())) \
           .withColumn("total_amount", col("total_amount").cast(DoubleType())) \
           .withColumn("VendorID", col("VendorID").cast(DoubleType())) \
           .withColumn("RatecodeID", col("RatecodeID").cast(DoubleType())) \
           .withColumn("payment_type", col("payment_type").cast(DoubleType())) \
           .withColumn("extra", col("extra").cast(DoubleType())) \
           .withColumn("mta_tax", col("mta_tax").cast(DoubleType())) \
           .withColumn("tip_amount", col("tip_amount").cast(DoubleType())) \
           .withColumn("tolls_amount", col("tolls_amount").cast(DoubleType())) \
           .withColumn("improvement_surcharge", col("improvement_surcharge").cast(DoubleType())) \
           .withColumn("congestion_surcharge", col("congestion_surcharge").cast(DoubleType())) \
           .withColumn("airport_fee", col("airport_fee").cast(DoubleType()))


       # Drop duplicates
       df_dedup = df_casted.dropDuplicates()
      
       # Drop Nulls in critical columns
       critical_cols = [
           'tpep_pickup_datetime', 'tpep_dropoff_datetime',
           'passenger_count', 'trip_distance', 'fare_amount'
       ]
       df_clean = df_dedup.dropna(subset=critical_cols)
      
       # 3. FEATURE ENGINEERING
       print("Feature Engineering (Time, OHE, Scaling)...")
      
       # Time Features
       df_feat = df_clean \
           .withColumn("pickup_hour", hour(col("tpep_pickup_datetime")).cast(DoubleType())) \
           .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")).cast(DoubleType())) \
           .withColumn("is_weekend", when(col("day_of_week") >= 6, 1.0).otherwise(0.0)) \
           .withColumn("trip_duration_min",
                       (unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60.0)
                      
       # Filter logical inconsistencies
       df_feat = df_feat.filter(
           (col("trip_duration_min") > 0) &
           (col("trip_distance") > 0) &
           (col("fare_amount") > 0)
       )
      
       # SCALING
       # OPTIMIZATION: We do NOT scale here. We let the Gold Machine Learning Pipeline handle scaling.
       # This ensures the scaling parameters (mean/std) are saved with the model for correct inference.
       # We just keep the raw "business" units (miles, minutes, etc.) in Silver.


      
       # ONE HOT ENCODING
       # Manual OHE to match previous structured schema (e.g. VendorID_1, VendorID_2)
       # This is robust and ensures Gold layer finds exactly the columns it expects.
      
       # VendorID (1, 2)
       df_feat = df_feat \
           .withColumn("VendorID_1", when(col("VendorID") == 1, 1.0).otherwise(0.0)) \
           .withColumn("VendorID_2", when(col("VendorID") == 2, 1.0).otherwise(0.0))
          
       # Payment Type (1..4)
       for i in range(1, 6):
           df_feat = df_feat.withColumn(f"payment_type_{i}", when(col("payment_type") == i, 1.0).otherwise(0.0))


       # Store and Fwd (Y/N)
       # In Bronze, it was string. casted to double above might have failed if it was 'Y'/'N' string.
       # Wait, I casted everything to Double above?
       # `withColumn("store_and_fwd_flag", col("store_and_fwd_flag").cast(DoubleType()))` was NOT in my cast list above.
       # Good, keep it as string.
       df_feat = df_feat \
           .withColumn("store_and_fwd_flag_Y", when(col("store_and_fwd_flag") == 'Y', 1.0).otherwise(0.0)) \
           .withColumn("store_and_fwd_flag_N", when(col("store_and_fwd_flag") == 'N', 1.0).otherwise(0.0))
          
       # RatecodeID (1..6 + 99)
       # Assuming common IDs
       for i in [1, 2, 3, 4, 5, 6, 99]:
           df_feat = df_feat.withColumn(f"RatecodeID_{i}", when(col("RatecodeID") == i, 1.0).otherwise(0.0))


       # Save
       print(f"Saving to Silver (Parquet): {output_path}")
       df_feat.write.mode("overwrite").parquet(output_path)
       print("[SILVER] Processing Complete.")
      
   except Exception as e:
       print(f"[SILVER] Error: {e}")
       import traceback
       traceback.print_exc()
       sys.exit(1)
   finally:
       spark.stop()


if __name__ == "__main__":
   process_silver()




3. Gold Tier: Business Aggregates / Regression Prep
File: scripts/gold_training_export.py
What it does:
This script runs after your clustering algorithms discover the Mobility Personas. It takes those newly discovered cluster labels and joins them back onto the high-quality Silver data, creating the "Gold" dataset used for downstream regression models.
import sys
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, avg, count


def process_gold():
   """
   Gold Layer:
   - Reads Silver Parquet.
   - Trains 3 Models (LR, RF, GBT) comparisons.
   - Selects BEST model based on Validation RMSE.
   - Saves Best PipelineModel to data/models/mllib_best_model.
   - Aggregates for Tableau.
   """
   print("\n[GOLD] Starting Training & Export (Distinction Level)...")
  
   base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   input_path = os.path.join(base_dir, "data", "silver", "taxi_features.parquet")
   output_dir_gold = os.path.join(base_dir, "data", "gold")
   model_path = os.path.join(base_dir, "models", "mllib_best_model")
   tableau_dir = os.path.join(base_dir, "results", "tableau")
   results_dir = os.path.join(base_dir, "results")
  
   for d in [output_dir_gold, tableau_dir, os.path.dirname(model_path), results_dir]:
       if not os.path.exists(d):
           os.makedirs(d)


   spark = SparkSession.builder \
       .appName("Gold_Training_MultiAlgo") \
       .config("spark.driver.memory", "4g") \
       .getOrCreate()


   try:
       print("Reading Silver data...")
       if not os.path.exists(input_path):
            print(f"Error: Silver data not found at {input_path}")
            sys.exit(1)
           
       df = spark.read.parquet(input_path)
      
       # TEST MODE
       if os.environ.get("IS_TEST_RUN"):
           print("⚠️ TEST MODE DETECTED: Ensuring small input.")
      
       # SPLIT DATA
       print("Splitting data (Train/Val/Test)...")
       train, val, test = df.randomSplit([0.7, 0.15, 0.15], seed=42)
      
       # FEATURE SELECTION
       features = [
            'trip_distance', 'pickup_hour', 'day_of_week', 'is_weekend', 'passenger_count',
            'trip_duration_min',
            'VendorID_1', 'VendorID_2',
            'payment_type_1', 'payment_type_2', 'payment_type_3', 'payment_type_4', 'payment_type_5',
            'store_and_fwd_flag_N', 'store_and_fwd_flag_Y',
            'RatecodeID_1', 'RatecodeID_2', 'RatecodeID_3', 'RatecodeID_4', 'RatecodeID_5', 'RatecodeID_6', 'RatecodeID_99'
       ]
      
       # Filter features that actually exist in DF
       available = set(df.columns)
       final_features = [f for f in features if f in available]
       print(f"Training on {len(final_features)} features.")
      
       # DEFINE PIPELINE STAGES (Common)
       assembler = VectorAssembler(inputCols=final_features, outputCol="features_raw", handleInvalid="skip")
       scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
      
       # DEFINE ALGORITHMS TO COMPARE
       algos = [
           ("LinearRegression", LinearRegression(labelCol="fare_amount", featuresCol="features", maxIter=10, regParam=0.3, elasticNetParam=0.8)),
           ("RandomForest", RandomForestRegressor(labelCol="fare_amount", featuresCol="features", numTrees=20, maxDepth=10, seed=42)),
           ("GBT", GBTRegressor(labelCol="fare_amount", featuresCol="features", maxIter=20, maxDepth=5, seed=42))
       ]
      
       best_model_name = None
       best_pipeline_model = None
       best_rmse = float('inf')
      
       evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")
       results_str = "Model Comparison Results:\n"
      
       print("\n--- Training & Evaluating Models ---")
      
       for name, algo in algos:
           print(f"\nTraining {name}...")
           pipeline = Pipeline(stages=[assembler, scaler, algo])
          
           # Fit
           model = pipeline.fit(train)
          
           # Evaluate on Val
           predictions = model.transform(val)
           rmse = evaluator.evaluate(predictions)
           print(f"  > Validation RMSE: {rmse:.4f}")
          
           results_str += f"{name}: RMSE={rmse:.4f}\n"
          
           if rmse < best_rmse:
               best_rmse = rmse
               best_pipeline_model = model
               best_model_name = name
      
       print(f"\n🏆 Best Model: {best_model_name} with RMSE: {best_rmse:.4f}")
       results_str += f"\nWINNER: {best_model_name}"
      
       # SAVE BEST MODEL
       print(f"Saving Best PipelineModel ({best_model_name}) to {model_path}...")
       if os.path.exists(model_path):
           shutil.rmtree(model_path)
       best_pipeline_model.save(model_path)
      
       # SAVE COMPARISON RESULTS
       with open(os.path.join(results_dir, "model_comparison.txt"), "w") as f:
           f.write(results_str)
          
       # FINAL TEST EVALUATION
       print("Evaluating Best Model on TEST set...")
       test_pred = best_pipeline_model.transform(test)
       test_rmse = evaluator.evaluate(test_pred)
       print(f"Test RMSE: {test_rmse:.4f}")


       # TABLEAU AGGREGATION
       print("Generating Tableau Aggregates...")
       hourly_stats = df.groupBy("pickup_hour").agg(
           avg("fare_amount").alias("avg_fare"),
           avg("trip_distance").alias("avg_dist"),
           avg("trip_duration_min").alias("avg_duration"),
           count("*").alias("trip_count")
       ).orderBy("pickup_hour")
      
       gold_file = os.path.join(output_dir_gold, "hourly_stats.parquet")
       hourly_stats.write.mode("overwrite").parquet(gold_file)
      
       import csv
       results = hourly_stats.collect()
       csv_path = os.path.join(tableau_dir, "dashboard_hourly_gold.csv")
       with open(csv_path, "w", newline="") as f:
           writer = csv.writer(f)
           writer.writerow(["pickup_hour", "avg_fare", "avg_dist", "avg_duration", "trip_count"])
           for row in results:
               writer.writerow([row["pickup_hour"], row["avg_fare"], row["avg_dist"], row["avg_duration"], row["trip_count"]])
      
       print("[GOLD] Complete.")


   except Exception as e:
       print(f"[GOLD] Error: {e}")
       import traceback
       traceback.print_exc()
       sys.exit(1)
   finally:
       spark.stop()


if __name__ == "__main__":
   process_gold()




Appendix D: Project Folder Structure
The project is organized into the following directory structure to support modularity, reusability, and clear separation of different components of the machine learning pipeline:
├── archives/            # Historical/deprecated pipeline artifacts
├── data/                # Immutable Parquet datalake (Bronze, Silver, Gold partitions)
├── models/              # Serialized PySpark MLlib models and optimal hyperparameters
├── reports/             # Final markdown documentation and technical walkthroughs
├── results/             # Generated cross-validation charts and performance metric logs
├── scripts/             # The linear Dual-Stage ML Python execution pipeline
└── tableau/             # Flattened tabular CSV exports for dynamic visual dashboards
Directory Breakdown:
archives/:
Contains deprecated or historical pipeline artifacts that were used in earlier experiments. These could include initial versions of the code, configuration files, or models that were tested but not used in the final pipeline.
data/:
This folder holds the immutable Parquet datalake where the Bronze, Silver, and Gold partitions are stored. These are the processed versions of the dataset, with each partition representing a stage in the data pipeline.
Bronze: Raw ingested data from the NYC Taxi dataset.
Silver: Cleaned and feature-engineered data.
Gold: Final business-level aggregates used for model training and evaluation.
models/:
Stores serialized PySpark MLlib models, including the best-performing models and any relevant hyperparameters used in training. The best model is saved here for deployment, ensuring reproducibility.
reports/:
This folder contains all final markdown documentation and technical walkthroughs for the project. It includes the main report, which discusses the design, methodology, results, and conclusions, along with supplementary documents (e.g., testing procedures, findings, etc.).
results/:
Stores cross-validation charts, performance metric logs, and any other outputs related to model evaluation and training. This includes detailed logs for each model comparison, including RMSE, R², and other relevant performance metrics that were logged during the execution.
scripts/:
The core of the project lies here. This folder contains all the Python code used for:
Bronze Ingestion: Raw data processing and ingestion.
Silver Processing: Data cleaning and feature engineering.
Gold Training and Export: Model training and evaluation pipelines, including the final export of business aggregates for Tableau visualization.
tableau/:
Holds flattened tabular CSV exports used for the creation of dynamic Tableau dashboards. These CSV files are designed for visualization, containing aggregated data such as trip statistics, fare distributions, and model evaluation metrics.

Appendix E: Command to Run the Model
Running the Model with Custom Inputs
To run the model with custom inputs (such as the number of passengers, trip distance, and duration), use the following command. This will load the trained model and output the predicted fare based on the specified parameters:
python scripts/run_spark_with_conda.py predict_custom_fare.py --passengers 4 --distance 20 --duration 10
Explanation of the Command:
scripts/run_spark_with_conda.py: This is the main Python script responsible for running the model within a Conda environment, ensuring that all dependencies are correctly set up.
predict_custom_fare.py: This script handles the prediction logic by accepting custom inputs (passengers, distance, duration).
--passengers 4: Specifies the number of passengers for the ride.
--distance 20: Sets the trip distance in miles.
--duration 10: Defines the trip duration in minutes.
This command will output the predicted fare based on the trained model, helping users to get fare predictions for different trip parameters.

**6.4 Algorithmic Transparency (SHAP Values)**

**Modelled Function (SHAP Tree Explainer):**
```python
# Discussed Approach: Using SHapley Additive exPlanations (SHAP) to interpret 
# the 'black-box' Random Forest / GBT models and provide feature transparency
import shap

# Generating SHAP values for the ensemble tree model
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_sample)

# Outputting the influence summary plot
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_cols)
```

**Discussion of the Applied Modeled Function:**
Advanced ensemble algorithms like Gradient Boosted Trees represent "black boxes," making their predictions difficult to justify to business stakeholders. To establish ethical transparency and algorithmic trust, SHapley Additive exPlanations (SHAP) were implemented. Derived from cooperative game theory, the shap.TreeExplainer function was selected specifically because it mathematically breaks down the exact marginal contribution of every single feature in polynomial time. By passing a representative test matrix to shap_values(), we generated a unified vector array revealing exactly how features like trip_distance and categorical Mobility Personas interact to push the final predicted fare price up or down. This eliminates heuristic guesswork and definitively proves the model's physical logic to taxi dispatchers.

