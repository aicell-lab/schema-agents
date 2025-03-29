# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Function to plot heatmap
def plot_heatmap(differential_expression_results):
    # Assuming 'differential_expression_results' is a pandas DataFrame with gene expression data
    heatmap_figure = sns.heatmap(differential_expression_results, cmap='viridis')
    plt.title('Gene Expression Heatmap')
    plt.xlabel('Samples')
    plt.ylabel('Genes')
    plt.show()
    return heatmap_figure

# Function to plot volcano plot
def plot_volcano(differential_expression_results):
    # Assuming 'differential_expression_results' contains columns 'fold_change' and 'p_value'
    volcano_figure = plt.scatter(differential_expression_results['fold_change'], -np.log10(differential_expression_results['p_value']))
    plt.title('Volcano Plot')
    plt.xlabel('Fold Change')
    plt.ylabel('-log10(p-value)')
    plt.show()
    return volcano_figure
