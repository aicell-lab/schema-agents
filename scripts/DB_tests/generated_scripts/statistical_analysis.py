# Import necessary libraries
import pandas as pd
from rpy2.robjects import r, pandas2ri

# Enable the conversion between pandas dataframes and R dataframes
pandas2ri.activate()

# Define the function to perform differential expression analysis
def perform_differential_expression(normalized_data):
    # Convert the pandas DataFrame to an R DataFrame
    r_data = pandas2ri.py2rpy(normalized_data)

    # Load the DESeq2 library
    r('library(DESeq2)')

    # Create a DESeq2 object from the R DataFrame
    deseq_dataset = r('DESeqDataSetFromMatrix(countData = r_data, colData = DataFrame(condition=colnames(r_data)), design= ~ condition)')

    # Run the DESeq2 algorithm
    deseq_results = r('results(DESeq(deseq_dataset))')

    # Convert the results back to a pandas DataFrame
    differential_expression_results = pandas2ri.rpy2py(deseq_results)

    # Return the results
    return differential_expression_results
