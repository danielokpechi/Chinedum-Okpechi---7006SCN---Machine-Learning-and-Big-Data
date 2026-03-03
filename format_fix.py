import re

with open("reports/project_report.md", "r") as f:
    text = f.read()

# 1. Bronze Tier Code Block
text = re.sub(
    r'(Modelled Function \(RDD Parallel Ingestion & Schema Enforcing\):)\s*(# Discussed Approach: Bypassing standard DataFrame.*?df = spark\.createDataFrame\(data_rdd, schema=schema\))\s*(Data ingestion was explicitly engineered)',
    r'**\1**\n```python\n\2\n```\n\n**Discussion of the Applied Modeled Function:**\n\3',
    text,
    flags=re.DOTALL
)

# 2. EDA Univariate
text = re.sub(
    r'(Modelled Function \(Distribution Profiling\):)\s*(# Extracting the 1% and 99% quantiles.*?df_filtered = df\.filter\(\(col\("fare_amount"\) >= lower_bound\) & \(col\("fare_amount"\) <= upper_bound\)\))\s*(The approxQuantile function is a highly optimized)',
    r'**\1**\n```python\n\2\n```\n\n**Discussion of the Applied Modeled Function:**\n\3',
    text,
    flags=re.DOTALL
)

# 3. EDA Bivariate
text = re.sub(
    r'(Modelled Function \(Correlation Matrix\):)\s*(# Assembling numerical features into a single Vector to.*?print\(pearson_matrix\.toArray\(\)\))\s*(Discussion of the Applied Modeled Function:)',
    r'**\1**\n```python\n\2\n```\n\n**\3**',
    text,
    flags=re.DOTALL
)

# 4. Gold Tier Aggregations
text = re.sub(
    r'(Modelled Function \(Distributed GroupBy Aggregations\):)\s*(# Discussed Approach: Using PySpark SQL functions.*?\.orderBy\("pickup_hour"\))\s*Discussion of the Applied Modeled Function:\s*To synthesize 37 million records into actionable dashboard metrics, the groupBy\(\)\.agg\(\) function was modeled\..*?rapidly for business dashboard exports\.\s*Discussion of the Applied Modeled Function:\s*(To synthesize 37 million records)',
    r'**\1**\n```python\n\2\n```\n\n**Discussion of the Applied Modeled Function:**\n\3',
    text,
    flags=re.DOTALL
)

# 5. ML Pipeline
text = re.sub(
    r'(# Discussed Approach: Vectorizing features and combining them with the.*?model = pipeline\.fit\(train\))\s*(Supervised modeling was executed utilizing PySpark\'s Pipeline object)',
    r'**Modelled Function (MLlib Algorithm Pipeline):**\n```python\n\1\n```\n\n**Discussion of the Applied Modeled Function:**\n\2',
    text,
    flags=re.DOTALL
)


# 6. Evaluation Bullet Points
text = re.sub(
    r'(The following results were observed for each model:)\s*Gradient Boosted Trees \(GBT\):\s*RMSE \(Test\): (\$8\.08)\s*R²: (0\.82)\s*MAE: (\$2\.16)\s*Linear Regression \(LR\):\s*RMSE \(Test\): (\$12\.45)\s*R²: (0\.60)\s*MAE: (\$4\.02)\s*Random Forest \(RF\):\s*RMSE \(Test\): (\$10\.36)\s*R²: (0\.72)\s*MAE: (\$3\.25)',
    r'\1\n\n- **Gradient Boosted Trees (GBT):**\n  - RMSE (Test): \2\n  - R²: \3\n  - MAE: \4\n- **Linear Regression (LR):**\n  - RMSE (Test): \5\n  - R²: \6\n  - MAE: \7\n- **Random Forest (RF):**\n  - RMSE (Test): \8\n  - R²: \9\n  - MAE: \10',
    text,
    flags=re.DOTALL
)

# 7. Model Comparison Table (The "table of numbers")
text = re.sub(
    r'Model\s*RMSE \(Test\)\s*R²\s*MAE\s*Gradient Boosted Trees\s*\$8\.08\s*0\.82\s*\$2\.16\s*Linear Regression\s*\$12\.45\s*0\.60\s*\$4\.02\s*Random Forest\s*\$10\.36\s*0\.72\s*\$3\.25',
    r'| Model | RMSE (Test) | R² | MAE |\n|---|---|---|---|\n| Gradient Boosted Trees | $8.08 | 0.82 | $2.16 |\n| Linear Regression | $12.45 | 0.60 | $4.02 |\n| Random Forest | $10.36 | 0.72 | $3.25 |',
    text,
    flags=re.DOTALL
)

# 8. SHAP Tree Explainer
text = re.sub(
    r'(Algorithmic Transparency \(SHAP Values\)Modelled Function \(SHAP Tree Explainer\):)\s*(# Discussed Approach: Using SHapley Additive exPlanations.*?shap\.summary_plot\(shap_values, X_test_sample, feature_names=feature_cols\))\s*(Advanced ensemble algorithms like Gradient Boosted Trees)',
    r'**6.4 Algorithmic Transparency (SHAP Values)**\n\n**Modelled Function (SHAP Tree Explainer):**\n```python\n\2\n```\n\n**Discussion of the Applied Modeled Function:**\n\3',
    text,
    flags=re.DOTALL
)

with open("reports/project_report.md", "w") as f:
    f.write(text)

