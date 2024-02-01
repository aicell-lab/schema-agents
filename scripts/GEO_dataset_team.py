import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
import requests
from xml.etree import ElementTree as ET
import re
from collections import defaultdict
from typing import Union

class OverallPlan(BaseModel):
    """The plan for how to analyze the NCBI GEO series given an input _series_matrix.txt file."""
    data_overview: str = Field(..., description="A description of the data: what files are available, how they were generated, and what they mean.")
    analysis_steps: str = Field(..., description="A detailed description of the analysis steps to process the data.")

class InformaticsPlanDraft(BaseModel):
    """The detailed informatics comprehensive analysis plan for how to analyze the NCBI GEO series data from start (package installation) to finish."""
    informatics_steps: list[str] = Field(..., description="A list of the informatics steps to process the data.")

class InformaticsPlan(InformaticsPlanDraft):
    """The detailed informatics comprehensive analysis plan for how to analyze the NCBI GEO series data from start (package installation) to finish."""
    # informatics_steps: list[dict] = Field(..., description="A list of the informatics steps to process the data.")

class Dependencies(BaseModel):
    """All of the software dependencies for the informatics comprehensive analysis plan."""
    dependencies: str = Field(..., description="The dependencies for the informatics comprehensive analysis plan in the format of a conda environment file.")

class InformaticsWorkflow(BaseModel):
    """The entire executable workflow for how to analyze the NCBI GEO series data from start (package installation) to finish in the form of a Snakemake workflow."""
    workflow: str = Field(..., description="The entire executable workflow for how to analyze the NCBI GEO series data from start (package installation) to finish in the form of a Snakemake workflow.")

async def make_overall_plan(matrix_file_path: str, role: Role = None) -> OverallPlan:
    """Make the overall plan for how to analyze the NCBI GEO series given an input _series_matrix.txt file."""
    with open(matrix_file_path, "r") as f:
        matrix_file_contents = f.read()
    result = await role.aask(matrix_file_contents, OverallPlan)
    return(result)   

async def make_informatics_plan_draft(overall_plan: OverallPlan, role: Role = None) -> InformaticsPlanDraft:
    """Make the informatics comprehensive analysis plan for how to analyze the NCBI GEO series given an input _series_matrix.txt file."""
    result = await role.aask(overall_plan, InformaticsPlanDraft)
    return(result)

async def make_informatics_plan(informatics_plan_draft: InformaticsPlanDraft, role: Role = None) -> InformaticsPlan:
    """Take the informatics draft plan and revise to create a completely self-contained bioinformatics workflow if necessary ensuring it meets the following criteria: (1) It is completely self-contained (2) it can be run without errors from start to finish"""
    result = await role.aask(informatics_plan_draft, InformaticsPlan)
    return(result)

async def internal_make_dependencies(informatics_plan: InformaticsPlan, role: Role = None) -> Dependencies:
    """Take the informatics plan and make a list of all of the software dependencies."""
    
    dependency_checker = Role(
        name="dependency_checker",
        profile="The dependency checker responsible for checking the dependencies for the informatics comprehensive analysis plan for how to analyze the NCBI GEO series data",
        goal="To check and list out dependencies for the informatics comprehensive analysis plan for how to analyze the NCBI GEO series data",
        constraints=None,
        actions=[],
    )

    result = await dependency_checker.aask(informatics_plan, Dependencies)
    return(result)

async def make_informatics_workflow(informatics_plan: InformaticsPlan, role: Role = None) -> InformaticsWorkflow:
    """Take the informatics plan and make a detailed step-by-step informatics workflow."""

    dependencies = await internal_make_dependencies(informatics_plan)
    result = await role.aask((informatics_plan, dependencies), InformaticsWorkflow)
    return(result)

async def make_snakemake_file(informatics_workflow: InformaticsWorkflow, role: Role = None) -> str:
    """Take the informatics workflow and make a Snakemake file."""
    result = await role.aask(informatics_workflow, str)
    with open("Snakefile", "w") as f:
        f.write(result)
    return(result)



# Main function
async def main():
    agents = []
    dataset_director = Role(
        name="dataset_director",
        profile="The dataset director responsible for directing the analysis of a GEO dataset",
        goal="To create a comprehensive analysis plan for how to analyze the NCBI GEO series data",
        constraints=None,
        actions=[make_overall_plan],
    )
    agents.append(dataset_director)

    informatics_planner = Role(
        name="informatics_planner",
        profile="The informatics planner responsible for drafting the informatics comprehensive analysis plan for how to analyze the NCBI GEO series data",
        goal="To create the informatics comprehensive analysis plan for how to analyze the NCBI GEO series data",
        constraints=None,
        actions=[make_informatics_plan_draft],
    )
    agents.append(informatics_planner)

    informatics_checker = Role(
        name="informatics_checker",
        profile="The informatics checker responsible for checking the informatics comprehensive analysis plan for how to analyze the NCBI GEO series data",
        goal="To check and revise the informatics comprehensive analysis plan for how to analyze the NCBI GEO series data",
        constraints=None,
        actions=[make_informatics_plan],
    )
    agents.append(informatics_checker)

    workflow_drafter = Role(
        name="workflow_drafter",
        profile="The workflow drafter responsible for drafting the informatics comprehensive analysis plan for how to analyze the NCBI GEO series data",
        goal="To create the workflow for how to analyze the NCBI GEO series data",
        constraints=None,
        actions=[make_informatics_workflow],
    )
    agents.append(workflow_drafter)

    workflow_writer = Role(
        name="workflow_writer",
        profile="The workflow writer responsible for writing the workflow to an executable snakemake file",
        goal="To write the workflow to an executable snakemake file",
        constraints=None,
        actions=[make_snakemake_file],
    )
    agents.append(workflow_writer)

    

    team = Team(name="Full dataset analyzers",
                profile="A team of agents meant to take a dataset description and write a comprehensive error-free complete workflow to analyze the data",
                investment=0.7)
    team.hire(agents)
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
    user_request = "/Users/gkreder/Downloads/2024-02-01_exponential_chain/GSE254364/matrix/GSE254364_series_matrix.txt"
    responses = await team.handle(Message(content=user_request, role="User"))
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()