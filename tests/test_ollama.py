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
async def test_respond_ollama():
    role = Role(instructions="You are Bob, a software engineer, please write a Software Requirement Document.", 
                backend="ollama")
    responses = await role.aask("Write hello world in Python", SoftwareRequirementDocument)
    assert responses
