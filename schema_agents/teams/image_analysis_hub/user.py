import os
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import asyncio
from schema_agents.role import Action, Role


class UserRequirement(BaseModel):
    """User Requirement for Project Manager"""
    task_description: str = Field(..., description="A concise description of the task or feature that the user wants to achieve or implement.")
    input_data_type: Optional[str] = Field(default=None, description="Type of input data that the task will be working with, if applicable (e.g., images, text, etc.).")
    desired_output: Optional[str] = Field(default=None, description="Brief description of the desired output or result, if specific.")
    test_input_data: Optional[str] = Field(default=None, description="Example input data that can be used for testing the task or feature.")
    test_desired_output: Optional[str] = Field(default=None, description="Described the desired output or result for the test input data, if applicable.")
    # preferred_interaction: Optional[str] = Field(default=None, description="Preferred user interaction pattern if any, such as drag-and-drop, button clicks, etc.")
    additional_notes: Optional[str] = Field(default=None, description="Any additional notes, preferences, or special requirements that the user may have provided.")

async def create_software_requirements(req: str, role: Role) -> UserRequirement:
    """Create software requirement."""
    return await role.aask(req, UserRequirement)

User = Role.create(name="Jane",
                         profile="User",
                         goal="Load an image file and segment the cells in the image, count the cells then show the result image.",
                         constraints=None,
                         actions=[UserRequirement])
