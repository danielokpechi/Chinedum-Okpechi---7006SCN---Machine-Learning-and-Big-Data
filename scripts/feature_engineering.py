import pandas as pd
import os
import sys
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

def process_features():
    """
    Performs feature engineering on the cleaned Taxi dataset.
    Uses a 2-pass incremental approach to handle large data.
    """
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "processed", "yellow_taxi_trip_2023.parquet")
    output_file = os.path.join(base_dir, "data", "processed", "yellow_taxi_trip_2023_features.parquet")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
        
    print("Starting Feature Engineering...")
    
    # Define Columns
    target_col = 'total_amount' # Keeps as is, but used for reference
    
    # Numerical to Scale
    scale_cols = [
        'trip_distance', 
        'fare_amount', 
        'extra', 
        'mta_tax', 
        'tip_amount', 
        'tolls_amount', 
        'improvement_surcharge', 
        'congestion_surcharge', 
        'airport_fee',
        'passenger_count'
    ]
    
    # Categorical to Encode
    cat_cols = [
        'VendorID', 
        'RatecodeID', 
        'payment_type', 
        'store_and_fwd_flag'
    ]
    
    # Initialize Transformers
    scaler = StandardScaler()
    
    # For OneHot, we need to know all categories upfront to ensure consistent columns across chunks.
    # We will collect unique values during Pass 1.
    cat_uniques = {col: set() for col in cat_cols}
    
    # Open Parquet File using PyArrow to stream batches
    parquet_file = pq.ParquetFile(input_file)
    
    print("\n" + "="*50)
    print("PASS 1: FITTING SCALER & LEARN CATEGORIES")
    print("="*50)
    
    total_rows = 0
    
    # Iterating row groups/batches
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=100_000)):
        df_chunk = batch.to_pandas()
        
        # Partial Fit Scaler
        scaler.partial_fit(df_chunk[scale_cols])
        
        # Collect Uniques for Categorical
        for col in cat_cols:
            cat_uniques[col].update(df_chunk[col].unique())
            
        total_rows += len(df_chunk)
        print(f"Pass 1: Processed {total_rows} rows...", end='\r')
        
    print(f"\nPass 1 Completed. Learned stats from {total_rows} rows.")
    print(f"Categories found: { {k: len(v) for k, v in cat_uniques.items()} }")
    
    # Setup Encoder with learned categories
    # handle_unknown='ignore' is safe, though we saw all categories in Pass 1.
    # sparse_output=False (dense) because we will concat with dense dataframe anyway
    categories_list = [sorted(list(cat_uniques[col])) for col in cat_cols]
    encoder = OneHotEncoder(categories=categories_list, sparse_output=False, handle_unknown='ignore')
    
    print("\n" + "="*50)
    print("PASS 2: TRANSFORMING & SAVING")
    print("="*50)
    
    writer = None
    processed_count = 0
    
    # Re-open iterator for Pass 2
    parquet_file = pq.ParquetFile(input_file)
    
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=100_000)):
        df_chunk = batch.to_pandas()
        
        # 1. Feature Extraction (Time)
        # Verify datetime columns are datetime type (Parquet should preserve this, but safety first)
        df_chunk['tpep_pickup_datetime'] = pd.to_datetime(df_chunk['tpep_pickup_datetime'])
        df_chunk['tpep_dropoff_datetime'] = pd.to_datetime(df_chunk['tpep_dropoff_datetime'])
        
        df_chunk['pickup_hour'] = df_chunk['tpep_pickup_datetime'].dt.hour
        df_chunk['day_of_week'] = df_chunk['tpep_pickup_datetime'].dt.dayofweek
        df_chunk['is_weekend'] = (df_chunk['day_of_week'] >= 5).astype(int)
        
        # Duration in minutes
        duration_td = df_chunk['tpep_dropoff_datetime'] - df_chunk['tpep_pickup_datetime']
        df_chunk['trip_duration_min'] = duration_td.dt.total_seconds() / 60.0
        
        # 2. Scaling
        # Transform returns numpy array, replace columns
        scaled_data = scaler.transform(df_chunk[scale_cols])
        # Create new column names for scaled versions? Or replace?
        # Typically replacing is cleaner for ML, but keeping originals might be useful for analysis.
        # User prompt example: "df[['fare_amount']] = scaler..." (Replacement)
        # We will REPLACE to keep schema lean, as this file is for Modeling.
        df_chunk[scale_cols] = scaled_data
        
        # 3. Encoding
        encoded_data = encoder.fit_transform(df_chunk[cat_cols]) # fit_transform with preset categories is effectively transform
        # Create column names
        encoded_feature_names = encoder.get_feature_names_out(cat_cols)
        
        # Create DataFrame from encoded data
        df_encoded = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=df_chunk.index)
        
        # Concatenate: Original (minus cat cols) + New Time Code + Encoded
        # Drop original cat cols (since we encoded them)
        df_final = pd.concat([
            df_chunk.drop(columns=cat_cols),
            df_encoded
        ], axis=1)
        
        # 4. Save
        table = pa.Table.from_pandas(df_final, preserve_index=False)
        
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
            
        writer.write_table(table)
        processed_count += len(df_final)
        print(f"Pass 2: Saved {processed_count} rows...", end='\r')

    if writer:
        writer.close()
        
    print(f"\nCompleted. Output saved to {output_file}")

if __name__ == "__main__":
    process_features()
