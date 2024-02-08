import pandas as pd
import numpy as np

# Load the data from CSV file
file_path = '/content/2000data.csv'
data = pd.read_csv(file_path)

# Function to generate 1 billion synthetic data samples
def generate_large_synthetic_data(data, num_samples=10000):
    # Ensure that num_samples is not less than the number of original data points
    if num_samples < len(data):
        raise ValueError("num_samples must be greater than the size of the original data")

    # Generate synthetic data
    synthetic_data = data.sample(n=num_samples, replace=True).reset_index(drop=True)
    return synthetic_data

# Generate synthetic data
augmented_data_synthetic = generate_large_synthetic_data(data, num_samples=10000)

# Save augmented data to a new CSV file
output_file = '/content/10000data.csv'  # Replace with your desired output file path
augmented_data_synthetic.to_csv(output_file, index=False)
