import os
import pandas as pd

# Function to identify and clean the CSV files
def identify_and_clean_csv(csv_file_path, cleaned_csv_file_path, zero_threshold=2):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Count zeros in each row
    zero_counts = df.apply(lambda x: (x == 0).sum(), axis=1)
    
    # Identify rows with more than the specified threshold of zeros
    outlier_indices = zero_counts[zero_counts > zero_threshold].index
    
    # Remove outlier rows
    cleaned_df = df.drop(outlier_indices)
    
    # Save the cleaned DataFrame to a new CSV file
    cleaned_df.to_csv(cleaned_csv_file_path, index=False)    
    
    return len(outlier_indices)

# The path to the directory containing the CSV files, inferred from the user's image
csv_directory_path = './data/1/parametres/'

# The path to the directory where cleaned CSV files will be saved
cleaned_csv_directory_path = './cleaned_csv/'

# Create the cleaned CSV directory if it does not exist
if not os.path.exists(cleaned_csv_directory_path):
    os.makedirs(cleaned_csv_directory_path)

# List of cleaned csv files and their outlier counts
cleaned_files_summary = []

# Loop through each file in the directory
for file in os.listdir(csv_directory_path):
    if file.endswith('.csv'):
        # Full path to the original and cleaned files
        original_csv_file_path = os.path.join(csv_directory_path, file)
        cleaned_csv_file_path = os.path.join(cleaned_csv_directory_path, file)
        
        # Clean the CSV file and get the number of removed rows
        removed_rows_count = identify_and_clean_csv(original_csv_file_path, cleaned_csv_file_path, zero_threshold=2)
        
        # Append the summary info
        cleaned_files_summary.append((file, removed_rows_count))

# Display the summary of cleaned files
for summary in cleaned_files_summary:
    print(f"File {summary[0]} had {summary[1]} outlier rows removed.")