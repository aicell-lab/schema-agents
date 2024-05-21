#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from schema_agents.role import Role


class SoftwareRequirementDocument(BaseModel):
    """Write Software Requirement Document."""
    original_requirements: str = Field(description="The polished complete original requirements")
    product_goals: List[str] = Field(description="Up to 3 clear, orthogonal product goals. If the requirement itself is simple, the goal should also be simple")
    user_stories: List[str] = Field(description="up to 5 scenario-based user stories, If the requirement itself is simple, the user stories should also be less")
    ui_design_draft: str = Field(description="Describe the elements and functions, also provide a simple style description and layout description.")
    anything_unclear: str = Field(None, description="Make clear here.")

@pytest.mark.asyncio
async def test_respond_gemini():
    role = Role(instructions="You are Bob, a software engineer, you have access to function SoftwareRequirementDocument.", 
                backend="gemini")
    responses = await role.aask(", write a Software Requirement Document for a simple software that prints 'Hello World' in Python, use function SoftwareRequirementDocument", SoftwareRequirementDocument)
    assert responses
