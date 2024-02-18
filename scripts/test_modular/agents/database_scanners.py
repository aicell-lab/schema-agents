import sys
import os
from pydantic import BaseModel, Field, ConfigDict
from schema_agents.role import Role
from schemas import *
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

class GEOScanner(Role):
    """An agent for scanning the NCBI GEO database"""
    def __init__(self, out_dir: str = "generated_files"):
        super().__init__(
            name="GEOScanner",
            profile="An agent for scanning the NCBI GEO database",
            goal="To scan the NCBI GEO database",
            constraints=None,
            actions=[self.scan_for_summary],
        )
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    async def get_data_information(self, faiss_index : FAISS) -> str:
        """Get information about the data"""
        data_information_docs = faiss_index.similarity_search("What data is available from the paper?", k=5)
        data_information_chunks = '\n\n################### PAPER SECTION ###################\n\n'.join([x.page_content for x in data_information_docs])
        return(data_information_chunks)

    async def scan_for_summary(self, pdf_path: str, role: Role = None) -> PaperSummary:
        """Scans a PDF paperfor a summary"""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
        data_information_docs = await self.get_data_information(faiss_index)
        sys.exit(data_information_docs)
        
        main_findings_docs = faiss_index.similarity_search("What are the main findings of the paper?", k=5)
        main_findings = await role.aask(main_findings_docs, MainFindings)
        computational_methods_docs = faiss_index.similarity_search("What are the computational analysis methods used in the paper?", k=5)
        computational_methods = await role.aask(computational_methods_docs, ComputationalMethods)
        experimental_methods_docs = faiss_index.similarity_search("What are the experimental methods used in the paper?", k=5)
        experimental_methods = await role.aask(experimental_methods_docs, ExperimentalMethods)
        samples_docs = faiss_index.similarity_search("What are the samples used in the paper?", k=5)
        samples = await role.aask(samples_docs, Samples)

        # computational_methods = faiss_index.similarity_search("What are the computational analysis methods used in the paper?", k=5)
        # experimental_methods = faiss_index.similarity_search("What are the experimental methods used in the paper?", k=5)
        # samples = faiss_index.similarity_search("What are the samples used in the paper?", k=5)
        result = PaperSummary(main_findings=main_findings, 
                            computational_methods=computational_methods, 
                            experimental_methods=experimental_methods, 
                            samples=samples)
        out_dir = "generated_files"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "paper_summary.md"), "w") as file:
            file.write(f"# Main Findings\n{result.main_findings}\n")
            file.write(f"# Computational Methods\n{result.computational_methods}\n")
            file.write(f"# Experimental Methods\n{result.experimental_methods}\n")
            file.write(f"# Samples\n{result.samples}\n")
        return(result)

        

    async def scan_webpage(self, url: str, role: Role = None) -> ScrapedGEOPage:
        """Scans a webpage for a database entry"""
        loader = AsyncHtmlLoader([url])
        docs = loader.load()
        content = docs[0]
        result = await role.aask(content, ScrapedGEOPage)
        return(result)

    async def scan(self, matrix_file_path: str, role: Role = None) -> GEODataSet:
        """Scans a database entry"""
        with open(matrix_file_path, "r") as file:
            matrix_file_contents = file.read()
        result = await role.aask(matrix_file_contents, GEODataSet)
        with open(os.path.join(self.out_dir, f"{result.geo_accession}.md"), "w") as file:
            file.write(f"# Summary\n{result.summary}\n")
            file.write(f"# Samples\n{result.samples_description}\n")
            file.write(f"# Data Generation\n{result.data_generation_summary}\n")
            file.write(f"# Data Files\n")
            for i_data_file, data_file in enumerate(result.data_files):
                file.write(f"## Data File {i_data_file}\n")
                file.write(f"Path : {data_file.path}\n")
                file.write(f"Description : {data_file.data_description}\n")
            
        return(result)
    
    async def propose_hypotheses(self, geo_data_set: GEODataSet, role: Role = None) -> InformaticHypotheses:
        """Propose hypotheses to test based on the data"""
        result = await role.aask(geo_data_set, InformaticHypotheses)
        
        for hypothesis in result.hypotheses:
                with open(os.path.join(self.out_dir, hypothesis.file_name), "w") as file:
                    file.write(f"# Hypothesis\n{hypothesis.hypothesis}\n")
                    file.write(f"# Workflow\n{hypothesis.test_workflow}\n")
                    file.write(f"# Available Samples\n{hypothesis.samples_description}\n")
                    file.write(f"# Available Data Files\n")
                    for i_data_file, data_file in enumerate(hypothesis.data_files):
                        file.write(f"## File {i_data_file}\n")
                        file.write(f"File Path : {data_file.path}\n")
                        file.write(f"Data Description : {data_file.data_description}\n")
        return(result)