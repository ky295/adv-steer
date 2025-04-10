import pandas as pd

def extract_goals(input_file_path, output_file_path, num_entries=100):
    """
    Extract the specified number of entries from the 'goal' column of a CSV file
    and save them to a new CSV file with just that column.
    
    Args:
        input_file_path (str): Path to the original CSV file
        output_file_path (str): Path where the new CSV file will be saved
        num_entries (int): Number of entries to extract (default: 100)
    """
    try:
        # Read the original CSV file
        df = pd.read_csv(input_file_path)
        
        # Verify the 'goal' column exists
        if 'goal' not in df.columns:
            raise ValueError("The 'goal' column does not exist in the CSV file")
        
        # Extract the specified number of entries from the 'goal' column
        # (or all entries if the file has fewer than requested)
        num_to_extract = min(num_entries, len(df))
        goals_df = df['goal'].head(num_to_extract).to_frame()
        
        # Save to a new CSV file
        goals_df.to_csv(output_file_path, index=False)
        
        print(f"Successfully extracted {num_to_extract} goals to {output_file_path}")
        
    except Exception as e:
        print(f"Error processing the CSV file: {e}")

extract_goals("dataset/raw/advbench.csv", "dataset/advbench_instruction.csv")

