import pandas as pd
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq

def clean_data():
    """
    Reads the raw Yellow Taxi Trip CSV in chunks, cleans it, and saves to Parquet.
    Cleaning steps:
    1. Parse dates.
    2. Drop rows with missing values in key columns.
    3. Drop duplicates (within chunk).
    4. Save to Parquet.
    """
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "raw", "yellow_taxi_trip_2023.csv")
    output_file = os.path.join(base_dir, "data", "processed", "yellow_taxi_trip_2023.parquet")
    
    # Check inputs
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
        
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # If output exists, remove it to start fresh (or we could error out)
    if os.path.exists(output_file):
        print(f"Removing existing output file: {output_file}")
        os.remove(output_file)
        
    print(f"Starting data cleaning process...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    CHUNK_SIZE = 100_000
    
    # Statistics trackers
    total_rows_read = 0
    total_rows_saved = 0
    total_dropped_na = 0
    total_dropped_dup = 0
    
    try:
        # Initialize Parquet writer
        writer = None
        
        # Stream chunks
        # Parse dates immediately on load to save a step, or do it after.
        # doing it after allows 'low_memory=False' to work better for other cols first.
        chunk_iterator = pd.read_csv(
            input_file, 
            chunksize=CHUNK_SIZE, 
            low_memory=False,
            dtype={
                'store_and_fwd_flag': 'str',
                'PULocationID': 'int32',
                'DOLocationID': 'int32',
                'payment_type': 'int64',
                'VendorID': 'int64'
            }
            # Note: We don't parse_dates here because invalid formats might crash the iterator.
            # We handle them inside the loop.
        )
        
        for i, chunk in enumerate(chunk_iterator):
            initial_len = len(chunk)
            total_rows_read += initial_len
            
            # 1. Convert Timestamps
            # errors='coerce' turns invalid parsing into NaT
            # Explicit format speeds up parsing significantly: 01/01/2023 12:32:10 AM -> %m/%d/%Y %I:%M:%S %p
            chunk['tpep_pickup_datetime'] = pd.to_datetime(chunk['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
            chunk['tpep_dropoff_datetime'] = pd.to_datetime(chunk['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
            
            # 2. Drop Missing Values
            # We focus on the columns identified as having missing values + key fields
            # Columns with missing: passenger_count, RatecodeID, store_and_fwd_flag, congestion_surcharge, airport_fee
            # Plus datetime cols if they became NaT
            subset_cols = [
                'tpep_pickup_datetime', 
                'tpep_dropoff_datetime', 
                'passenger_count', 
                'RatecodeID',
                'store_and_fwd_flag'
            ]
            
            chunk_cleaned = chunk.dropna(subset=subset_cols)
            
            # Enforce types to ensure schema consistency across chunks
            # After dropna, these should be safe to cast to int/str
            chunk_cleaned = chunk_cleaned.astype({
                'passenger_count': 'int64',
                'RatecodeID': 'int64',
                'VendorID': 'int64',
                'payment_type': 'int64',
                'PULocationID': 'int32',
                'DOLocationID': 'int32',
                'store_and_fwd_flag': 'str',
                'fare_amount': 'float64',
                'extra': 'float64',
                'mta_tax': 'float64',
                'tip_amount': 'float64',
                'tolls_amount': 'float64',
                'improvement_surcharge': 'float64',
                'total_amount': 'float64',
                'congestion_surcharge': 'float64',
                'airport_fee': 'float64',
                'trip_distance': 'float64'
            })

            rows_after_dna = len(chunk_cleaned)
            total_dropped_na += (initial_len - rows_after_dna)
            
            # 3. Drop Duplicates
            # Note: This checks duplicates ONLY within the current chunk.
            chunk_cleaned = chunk_cleaned.drop_duplicates()
            rows_after_dup = len(chunk_cleaned)
            total_dropped_dup += (rows_after_dna - rows_after_dup)
            
            if not chunk_cleaned.empty:
                # 4. Write to Parquet
                # preserve_index=False avoids __index_level_0__ mismatch issues between chunks
                table = pa.Table.from_pandas(chunk_cleaned, preserve_index=False)
                
                if writer is None:
                    # First chunk determines schema
                    writer = pq.ParquetWriter(output_file, table.schema)
                
                writer.write_table(table)
                total_rows_saved += len(chunk_cleaned)
            
            print(f"Processed chunk {i+1}: Read {initial_len}, Saved {len(chunk_cleaned)}...", end='\r')

        if writer:
            writer.close()
            
        print("\n" + "="*50)
        print("COMPLETED")
        print("="*50)
        print(f"Total Rows Read:    {total_rows_read}")
        print(f"Dropped (NaN):      {total_dropped_na}")
        print(f"Dropped (Dupes*):   {total_dropped_dup} (*chunk-level only)")
        print(f"Total Rows Saved:   {total_rows_saved}")
        print(f"Output saved to:    {output_file}")
        
    except Exception as e:
        print(f"\nAn error occurred during cleaning: {e}")
        # Clean up partial file
        if writer:
            writer.close()
        sys.exit(1)

if __name__ == "__main__":
    clean_data()
