#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List
from pydantic import BaseModel, Field
from schema_agents.role import Role

class SoftwareRequirementDocument(BaseModel):
    """Write Software Requirement Document."""
    original_requirements: str = Field(description="The polished complete original requirements")
    product_goals: List[str] = Field(description="Up to 3 clear, orthogonal product goals. If the requirement itself is simple, the goal should also be simple")
    user_stories: List[str] = Field(description="up to 5 scenario-based user stories, If the requirement itself is simple, the user stories should also be less")
    ui_design_draft: str = Field(description="Describe the elements and functions, also provide a simple style description and layout description.")
    anything_unclear: str = Field(None, description="Make clear here.")

class UserClarification(BaseModel):
    """Provide more details for the use case."""
    summary: str = Field(description="Summary of question to be clarified.")
    content: str = Field(description="Anwser to the clarification request.")

class GetExtraInformation(BaseModel):
    """Extra information needed to be able to work on the task."""
    content: str = Field(description="The information.")
    summary: str = Field(description="Summary of what you already get.")

BioImageAnalyst = Role.create(name="Alice",
            profile="BioImage Analyst",
            goal="Efficiently communicate with the user and translate the user's needs into software requirements",
            constraints=None,
            actions=[SoftwareRequirementDocument, GetExtraInformation],
            watch=[UserClarification])

User = Role.create(name="Bob",
            profile="User",
            goal="Provide the use case and requirements for the development, the aim is to create a cell counting software for U2OS cells under confocal microscope, I would like to use web interface in Imjoy.",
            constraints=None,
            actions=[UserClarification],
            watch=[GetExtraInformation])
