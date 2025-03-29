# Import necessary libraries
import pandas as pd
import json
import gzip

# Function to load gene expression count data
def load_expression_data(file_path: str) -> pd.DataFrame:
    with gzip.open(file_path, 'rt') as file:
        expression_data = pd.read_csv(file, sep='\t', header=None, comment='%')
    return expression_data

# Function to load normalization scale factors
def load_scale_factors(file_path: str) -> dict:
    with gzip.open(file_path, 'rt') as file:
        scale_factors = json.load(file)
    return scale_factors
