import os
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schemas import *

class GEOScanner(Role):
    """An agent for scanning the NCBI GEO database"""
    def __init__(self, out_dir: str = "generated_files"):
        super().__init__(
            name="GEOScanner",
            profile="An agent for scanning the NCBI GEO database",
            goal="To scan the NCBI GEO database",
            constraints=None,
            actions=[self.scan],
        )
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

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