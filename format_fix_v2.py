import re

with open("reports/project_report.md", "r") as f:
    text = f.read()

# 1. Page breaks for Abstract
text = re.sub(
    r'(Abstract\nThis study engineers.*?pricing, demand, and resource allocation\.\n)',
    r'\1\n<div style="page-break-after: always;"></div>\n',
    text,
    flags=re.DOTALL
)

# 2. Remove "Pandas"
text = re.sub(
    r'that a standard Pandas \.quantile\(\) call might trigger',
    r'that standard single-node memory-bound statistical operations might trigger',
    text
)


# 3. Bronze Tier Code Block
text = re.sub(
    r'(Modelled Function \(RDD Parallel Ingestion & Schema Enforcing\):)\n(# Discussed Approach: Bypassing standard DataFrame.*?df = spark\.createDataFrame\(data_rdd, schema=schema\))\n(Data ingestion was explicitly engineered)',
    r'**\1**\n```python\n\2\n```\n\n**Discussion of the Applied Modeled Function:**\n\3',
    text,
    flags=re.DOTALL
)

# 4. EDA Univariate
text = re.sub(
    r'(Modelled Function \(Distribution Profiling\):)\n(# Extracting the 1% and 99% quantiles.*?df_filtered = df\.filter\(\(col\("fare_amount"\) >= lower_bound\) & \(col\("fare_amount"\) <= upper_bound\)\))\n(The approxQuantile function is a highly optimized)',
    r'**\1**\n```python\n\2\n```\n\n**Discussion of the Applied Modeled Function:**\n\3',
    text,
    flags=re.DOTALL
)

# 5. EDA Bivariate
text = re.sub(
    r'(Modelled Function \(Correlation Matrix\):)\n(# Assembling numerical features into a single Vector to.*?print\(pearson_matrix\.toArray\(\)\))\n(Discussion of the Applied Modeled Function:)',
    r'**\1**\n```python\n\2\n```\n\n**\3**',
    text,
    flags=re.DOTALL
)

# 6. Gold Tier Aggregations
text = re.sub(
    r'(Modelled Function \(Distributed GroupBy Aggregations\):)\n(# Discussed Approach: Using PySpark SQL functions.*?\)\.orderBy\("pickup_hour"\))\nDiscussion of the Applied Modeled Function:\s*\nTo synthesize 37 million records into actionable dashboard metrics.*?(rapidly for business dashboard exports\.)Discussion of the Applied Modeled Function: To synthesize.*?rapidly for business dashboard exports\.',
    r'**\1**\n```python\n\2\n```\n\n**Discussion of the Applied Modeled Function:**\nTo synthesize 37 million records into actionable dashboard metrics, the `groupBy().agg()` function was modeled. In a distributed environment, standard grouping induces massive data shuffling across the network. By utilizing PySpark\'s native aggregation functions (`avg` and `count`), the Catalyst optimizer forces partial aggregations on each executor node *before* the network shuffle occurs (Map-Side Combine). This drastically reduces network bottlenecking, allowing us to compute macroscopic temporal demand trends (average fare, volume counts per hour) \3',
    text,
    flags=re.DOTALL
)


# 7. ML Pipeline
# Note: User's paste didn't have "Modelled Function (MLlib Algorithm Pipeline):" literally before the pipeline code block.
text = re.sub(
    r'(# Discussed Approach: Vectorizing features and combining them with the.*?model = pipeline\.fit\(train\))\n\n(Supervised modeling was executed utilizing PySpark\'s Pipeline object)',
    r'**Modelled Function (MLlib Algorithm Pipeline):**\n```python\n\1\n```\n\n**Discussion of the Applied Modeled Function:**\n\2',
    text,
    flags=re.DOTALL
)


# 8. Evaluation Bullet Points
text = re.sub(
    r'(The following results were observed for each model:)\nGradient Boosted Trees \(GBT\):\nRMSE \(Test\): (\$8\.08)\nR²: (0\.82)\nMAE: (\$2\.16)\nLinear Regression \(LR\):\nRMSE \(Test\): (\$12\.45)\nR²: (0\.60)\nMAE: (\$4\.02)\nRandom Forest \(RF\):\nRMSE \(Test\): (\$10\.36)\nR²: (0\.72)\nMAE: (\$3\.25)',
    r'\1\n\n- **Gradient Boosted Trees (GBT):**\n  - RMSE (Test): \2\n  - R²: \3\n  - MAE: \4\n- **Linear Regression (LR):**\n  - RMSE (Test): \5\n  - R²: \6\n  - MAE: \7\n- **Random Forest (RF):**\n  - RMSE (Test): \8\n  - R²: \9\n  - MAE: \10\n',
    text,
    flags=re.DOTALL
)

# 9. Model Comparison Table
text = re.sub(
    r'Model\nRMSE \(Test\)\nR²\nMAE\nGradient Boosted Trees\n\$8\.08\n0\.82\n\$2\.16\nLinear Regression\n\$12\.45\n0\.60\n\$4\.02\nRandom Forest\n\$10\.36\n0\.72\n\$3\.25',
    r'| Model | RMSE (Test) | R² | MAE |\n|---|---|---|---|\n| Gradient Boosted Trees | $8.08 | 0.82 | $2.16 |\n| Linear Regression | $12.45 | 0.60 | $4.02 |\n| Random Forest | $10.36 | 0.72 | $3.25 |',
    text,
    flags=re.DOTALL
)

# 10. SHAP Tree Explainer
text = re.sub(
    r'(Algorithmic Transparency \(SHAP Values\)Modelled Function \(SHAP Tree Explainer\):)\n(# Discussed Approach: Using SHapley Additive exPlanations.*?shap\.summary_plot\(shap_values, X_test_sample, feature_names=feature_cols\))\n\n(Advanced ensemble algorithms like Gradient Boosted Trees)',
    r'**6.4 Algorithmic Transparency (SHAP Values)**\n\n**Modelled Function (SHAP Tree Explainer):**\n```python\n\2\n```\n\n**Discussion of the Applied Modeled Function:**\n\3',
    text,
    flags=re.DOTALL
)

with open("reports/project_report.md", "w") as f:
    f.write(text)

