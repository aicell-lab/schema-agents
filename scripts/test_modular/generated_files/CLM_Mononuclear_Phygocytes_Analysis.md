# Hypothesis
Identification of unique gene expression profiles in mononuclear phagocytes within human colo-rectal liver metastasis tissues can reveal clinically relevant markers for diagnosis and treatment.
# Workflow
1. Download and extract the GSE254364_RAW.tar dataset.
2. Use the Scanpy package to preprocess the data, including quality control, normalization, and feature selection.
3. Perform dimensionality reduction (e.g., PCA, t-SNE) to visualize the data and identify clusters of mononuclear phagocytes.
4. Use differential gene expression analysis to compare the identified clusters and highlight genes that are uniquely expressed in mononuclear phagocytes within CLM tissues.
5. Validate the identified markers through literature review and comparison with known markers of mononuclear phagocytes.
6. Discuss the potential clinical relevance of the identified markers for diagnosis and treatment of colo-rectal liver metastasis.
# Available Samples
This study includes two samples, GSM8041061 and GSM8041062, both involving FFPE Spatial transcriptome sequencing of human colo-rectal liver metastasis (CLM) tissues. The samples were processed using high-throughput sequencing on a NextSeq 2000 platform, focusing on total RNA extracted from the tissues. The data processing involved converting raw sequencing data into fastqs, aligning them to the human reference genome GRCh38, and analyzing the filtered feature matrices individually using the Scanpy package.
# Available Data Files
## File 0
File Path : ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE254nnn/GSE254364/suppl/GSE254364_RAW.tar
Data Description : barcodes.tsv.gz: list of spatial barcodes
features.tsv.gz: list of gene Ids
matrix.mtx.gz: gene expression count data in Matrix Market Exchange Format
aligned_fiducials.jpg.gz: aligned fiducials of the tissue image
detected_tissue_image.jpg.gz: image of tissue and spots
scalefactors_json.json.gz: scalefactors in json format
tissue_hires_image.png.gz: hi-res image of tissue
tissue_lowres_image.png.gz: low-res image of tissue
tissue_positions_list.csv.gz: list of spatial barcodes and the coordinates specifying spots
