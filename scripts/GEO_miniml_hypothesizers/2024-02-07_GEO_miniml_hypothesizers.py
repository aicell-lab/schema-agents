import os
import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
from xml.etree import ElementTree as ET

class DataFile(BaseModel):
    """A file in the data repository coming from a sample"""
    name: str = Field(..., description="The name of the file")
    desc: str = Field(..., description="A description of the file")
    location : str = Field(..., description="The location of the file")
    format : str = Field(..., description="The format of the file")

class Sample(BaseModel):
    """A sample in the data repository"""
    desc: str = Field(..., description="A description of the sample")
    data_files: List[DataFile] = Field(..., description="A list of the data files associated with the sample")

class DataDescription(BaseModel):
    """A description of all the data available in the repository as demarcated by a <Supplementary-Data> tag inside of a <Sample> tag in the matrix file"""
    samples: list[Sample] = Field(..., description="A detailed description of the samples available in the repository")

class HypothesisWorkflow(BaseModel):
    """The workflow for testing a hypothesis"""
    hypothesis : str = Field(..., description="The hypothesis to test")
    workflow: list[str] = Field(..., description="The workflow for testing a hypothesis")
    samples: list[Sample]  = Field(..., description="The samples involved in this hypothesis workflow. This is a subset of the `samples` field in the data description model.")

class AnalysisOptions(BaseModel):
    """The possible hypotheses you could test using ONLY the data in the repository"""
    options: list[HypothesisWorkflow] = Field(..., description="All the possible hypotheses you could test and how to test them in pseudo-code format")

class WorkflowRequirements(BaseModel):
    """The requirements for a workflow"""
    file_name : str = Field(..., description="The name of the workflow file, it should contain no spaces, briefly refer to the hypothesis, and end in .md")
    data_files : list[DataFile] = Field(..., description="The data files required for the workflow. These are taken directly from the `samples` field in the HypothesisWorkflow model")
    analysis_steps : list[str] = Field(..., description="The analysis steps for the workflow")
    desired_output : str = Field(..., description="The desired output of the workflow")

def create_data_scanner():

    async def scan_matrix(matrix_path : str, role : Role = None) -> DataDescription:
        """Scans the matrix for available data"""
        with open(matrix_path, "r") as file:
            matrix_content = file.read()
        result = await role.aask(matrix_content, DataDescription)
        return(result)

    async def evaluate_analysis_options(data_description : DataDescription, role : Role = None) -> AnalysisOptions:
        """Evaluates the analysis options"""
        result = await role.aask(data_description, AnalysisOptions)
        for hypothesis in result.options:
            print(hypothesis)
        return(result)
    
    data_scanner = Role(
        name="Data Scanner",
        profile="An agent that scans for available data",
        goal="To comprehensively scan for available data",
        constraints=None,
        model = "gpt-4-0125-preview",
        actions=[scan_matrix, evaluate_analysis_options],
    )
    return(data_scanner)

def create_requirements_writer():

    def make_data_file_string(data_files : List[DataFile]) -> str:
        """Makes a string of the data files"""
        s = ""
        for df in data_files:
            s += f"Name : {df.name}\n"
            s += f"Description : {df.desc}\n"
            s += f"Location : {df.location}\n"
            s += f"Format : {df.format}\n\n"
        return s 
 
    async def write_requirements(analysis_options : AnalysisOptions, role : Role = None) -> None:
        """Writes the software requirements and specifications for all the analysis options"""
        tasks = [role.aask(hypothesis_workflow, WorkflowRequirements) for hypothesis_workflow in analysis_options.options]
        all_requirements = await asyncio.gather(*tasks)
        out_dir = "generated_files"
        os.makedirs(out_dir, exist_ok=True)
        for reqs in all_requirements:
            with open(os.path.join(out_dir, reqs.file_name), "w") as file:
                s = make_data_file_string(reqs.data_files)
                file.write(f"# Input Data Files\n{s}\n")
                s = '\n'.join([x for x in reqs.analysis_steps])
                file.write(f"# Analysis Steps\n{s}\n")
                file.write(f"# Desired Output\n{reqs.desired_output}\n")
        return
    
    requirements_writer = Role(
        name="Requirements Writer",
        profile="An agent that writes the software requirements for the hypotheses to investigate",
        goal="To write the requirements for the project",
        constraints=None,
        model = "gpt-4-0125-preview",
        actions=[write_requirements],
    )
    return(requirements_writer)

def make_team():
    agents = []
    data_scanner = create_data_scanner()
    agents.append(data_scanner)
    requirements_writer = create_requirements_writer()
    agents.append(requirements_writer)
    team = Team(name="NCBI GEO analyzers", profile="A team of agents meant to comprehensively understand and analyze a NCBI GEO repository", investment=0.7)
    team.hire(agents)
    return(team)

# Main function
async def main():
    team = make_team()
    user_request = "Test this"
    responses = await team.handle(Message(content=user_request, role="User"))
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()