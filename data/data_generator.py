# -*- coding: utf-8 -*-
"""data-generator.ipynb
"""

import pandas as pd
import numpy as np

file_path = '/content/full_data.txt'
data = pd.read_csv(file_path)

def generate_large_synthetic_data(data, num_samples=16000):
    if num_samples < len(data):
        raise ValueError("num_samples must be greater than the size of the original data")


    synthetic_data = data.sample(n=num_samples, replace=True).reset_index(drop=True)
    return synthetic_data

augmented_data_synthetic = generate_large_synthetic_data(data, num_samples=16000)

output_file = '/content/16kdata.txt'
augmented_data_synthetic.to_csv(output_file, index=False)