#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from schema_agents.teams import Team
from schema_agents.role import Role
from schema_agents.schema import Message

class SoftwareRequirementDocument(BaseModel):
    """Write Software Requirement Document."""
    product_goals: List[str] = Field(description="Up to 3 clear, orthogonal product goals. If the requirement itself is simple, the goal should also be simple")

class AnalysisScript(BaseModel):
    """Python script for image analysis."""
    script: str = Field(description="Python script for image analysis")


@pytest.mark.asyncio
async def test_team():
    async def create_user_requirements(query: str, role: Role) -> SoftwareRequirementDocument:
        """Create user requirements."""
        response = await role.aask(query, SoftwareRequirementDocument)
        return response
        
    bioimage_analyst = Role(name="Alice",
                profile="BioImage Analyst",
                goal="Efficiently communicate with the user and translate the user's needs into software requirements",
                constraints=None,
                actions=[create_user_requirements])
    
    async def write_code(req: SoftwareRequirementDocument, role: Role) -> AnalysisScript:
        """Write code for image analysis."""
        code = await role.aask(req, AnalysisScript)
        return code
    
    coder = Role(name="Bob",
                profile="Coder",
                goal="Write code for image analysis.",
                constraints=None,
                actions=[write_code])
    
    team = Team(name="Test Team")
    team.hire([bioimage_analyst, coder])
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
    responses = await team.handle(Message(role="Bot", content="Create a segmentation software"))
    assert isinstance(responses[1].data, SoftwareRequirementDocument)
    assert isinstance(responses[2].data, AnalysisScript)
    
    
