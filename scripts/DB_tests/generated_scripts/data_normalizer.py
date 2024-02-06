# This script contains functions to normalize gene expression data using scale factors.

import pandas as pd


def normalize_data(expression_data: pd.DataFrame, scale_factors: dict) -> pd.DataFrame:
    """
    Applies the scale factors to the gene expression data to normalize it for differential expression analysis.

    :param expression_data: DataFrame containing gene expression data.
    :param scale_factors: Dictionary with gene names as keys and scale factors as values.
    :return: DataFrame with normalized gene expression data.
    """
    normalized_data = expression_data.copy()
    for gene, scale in scale_factors.items():
        if gene in normalized_data.columns:
            normalized_data[gene] *= scale
    return normalized_data
