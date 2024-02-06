#!/usr/bin/env nextflow

// Workflow for Differential Gene Expression Analysis between CLM 01 and CLM 02 samples

params.data_clm01_tar_gz = ''
params.data_clm02_tar_gz = ''
params.output_dir = './results'

process DecompressData {
    input:
        path data_tar_gz from params.data_clm01_tar_gz, params.data_clm02_tar_gz
    output:
        path "*.csv" into decompressed_files
    script:
        "tar -xzf $data_tar_gz"
}

process NormalizeData {
    input:
        path data_csv from decompressed_files
    output:
        path "*.normalized.csv" into normalized_files
    script:
        "python -c 'import pandas as pd; from sklearn.preprocessing import normalize; data = pd.read_csv(\"$data_csv\"); data_normalized = normalize(data); data_normalized.to_csv(\"$data_csv.normalized.csv\", index=False)'"
}

process MergeNormalizedData {
    input:
        path data_normalized_csv from normalized_files.collect()
    output:
        path "combined_normalized_data.csv"
    script:
        "python -c 'import pandas as pd; import glob; files = glob.glob(\"*.normalized.csv\"); combined = pd.concat([pd.read_csv(f) for f in files]); combined.to_csv(\"combined_normalized_data.csv\", index=False)'"
}

process DifferentialExpressionAnalysis {
    input:
        path data_normalized_csv from MergeNormalizedData.out
    output:
        path "*.results.csv" into analysis_results
    script:
        "python -c 'from rpy2.robjects import r, pandas2ri; pandas2ri.activate(); r(\"library(DESeq2)\"); r(\"dds <- DESeqDataSetFromMatrix(countData = pd.read_csv(\"combined_normalized_data.csv\"), colData = <sample_information>, design = ~ sample)\"); r(\"dds = DESeq(dds)\"); r(\"res = results(dds)\"); r(\"resOrdered = res[order(res$padj),]\"); significant_genes = resOrdered[resOrdered[\'padj\'] < 0.05]; significant_genes.to_csv(\"$data_normalized_csv.results.csv\", index=False)'"
}

workflow {
    DecompressData()
    NormalizeData()
    MergeNormalizedData()
    DifferentialExpressionAnalysis()

    analysis_results.collect().set{ final_results }
    final_results.view { it -> "Results file: ${it}" }
}

// Note: This script assumes that the necessary Python and R libraries, along with their dependencies, are installed and configured in the execution environment. It also assumes that sample information for DESeq2 analysis is provided and formatted correctly.