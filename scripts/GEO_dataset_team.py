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

class InformaticsPlan(BaseModel):
    """The detailed informatics comprehensive analysis plan for how to analyze the NCBI GEO series data from start (package installation) to finish."""
    informatics_steps: list[dict] = Field(..., description="A list of the informatics steps to process the data.")

class Dependencies(BaseModel):
    """All of the software dependencies for the informatics comprehensive analysis plan."""
    dependencies: str = Field(..., description="The dependencies for the informatics comprehensive analysis plan in the format of a conda environment file.")

class InformaticsWorkflow(BaseModel):
    """The entire executable workflow for how to analyze the NCBI GEO series data from start (package installation) to finish in the form of a Snakemake workflow."""
    workflow: str = Field(..., description="The entire executable workflow for how to analyze the NCBI GEO series data from start (package installation) to finish in the form of a Snakemake workflow.")

async def make_overall_plan(matrix_file_path: str, role: Role = None) -> OverallPlan:
    """Make the overall plan for how to analyze the NCBI GEO series given an input _series_matrix.txt file."""
    result = await role.aask(user_input, OverallPlan)
    return(result)   

async def make_informatics_plan_draft(overall_plan: OverallPlan, role: Role = None) -> InformaticsPlanDraft:
    """Make the informatics comprehensive analysis plan for how to analyze the NCBI GEO series given an input _series_matrix.txt file."""
    result = await role.aask(user_input, InformaticsPlanDraft)
    return(result)

async def make_informatics_plan(informatics_plan_draft: InformaticsPlanDraft, role: Role = None) -> InformaticsPlan:
    """Take the informatics draft plan and ask, "is it completely self contained? Could I run this from start to finish?" If not, then ask, "what is missing?" and add it to the plan to create an entirely self-contained bioinformatic workflow."""
    result = await role.aask(user_input, InformaticsPlan)
    return(result)

async def make_dependencies(informatics_plan: InformaticsPlan, role: Role = None) -> Dependencies:
    """Take the informatics plan and make a list of all of the software dependencies."""
    result = await role.aask(user_input, Dependencies)
    return(result)

async def make_informatics_workflow(informatics_plan: InformaticsPlan, role: Role = None) -> InformaticsWorkflow:
    """Take the informatics plan and make a Snakemake workflow."""
    result = await role.aask(user_input, InformaticsWorkflow)
    return(result)