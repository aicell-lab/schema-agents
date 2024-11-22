import os
import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
from xml.etree import ElementTree as ET


class Dependency(BaseModel):
    """A software dependency"""
    name: str = Field(..., description="The name of the dependency")
    version: str = Field(..., description="The version of the dependency")

class Dependencies(BaseModel):
    """The software dependencies for the project listed exhaustively and precisely"""
    deps : list[Dependency]

class DataDownloadPlan(BaseModel):
    """The plan for how to download the data"""
    plan : str = Field(..., description="The plan for how to download the data")

class AnalysisPlan(BaseModel):
    """The plan for how to analyze the data"""
    plan : str = Field(..., description="The plan for how to analyze the data")

class ReportingPlan(BaseModel):
    """The plan for how to report the results of the analysis"""
    plan : str = Field(..., description="The plan for how to report the results of the analysis")

class WorkfowArchitecture(BaseModel):
    """The proposed architecture for the informatic workflow"""
    data_download_plan : DataDownloadPlan
    analysis_plan : AnalysisPlan
    reporting_plan : ReportingPlan

def create_project_architect():

    async def make_plan(user_input : str, role : Role = None) -> WorkfowArchitecture:
        """Makes the plan for the informatic workflow"""
        result = await role.aask(user_input, WorkfowArchitecture)
        return(result)
    
    project_architect = Role(
        name="Project Architect",
        profile="An agent that designs the architecture for a complete informatic workflow",
        goal="To design the architecture for a workflow",
        constraints=None,
        actions=[make_plan],
    )

    return(project_architect)

def make_team():
    agents = []
    project_architect = create_project_architect()
    agents.append(project_architect)

    team = Team(name="Informatic Workflow Designers", profile="A team of agents meant to design the architecture for a complete informatic workflow", investment=0.7)

    team.hire(agents)
    return(team)
