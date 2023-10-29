#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
import json


class SoftwareRequirementDocument(BaseModel):
    """Write Software Requirement Document."""
    original_requirements: str = Field(description="The polished complete original requirements")
    product_goals: List[str] = Field(description="Up to 3 clear, orthogonal product goals. If the requirement itself is simple, the goal should also be simple")
    user_stories: List[str] = Field(description="up to 5 scenario-based user stories, If the requirement itself is simple, the user stories should also be less")
    ui_design_draft: str = Field(description="Describe the elements and functions, also provide a simple style description and layout description.")
    anything_unclear: str = Field(None, description="Make clear here.")

class UserClarification(BaseModel):
    """Provide more details for the use case."""
    content: str = Field(description="Anwser to the clarification request.")

class SearchInternetQuery(BaseModel):
    """Keywords for searching the internet."""
    query: str = Field(description="space separated keywords for searching the internet.")

class GetExtraInformation(BaseModel):
    """Extra information needed to be able to work on the task."""
    content: str = Field(description="The information.")
    # summary: str = Field(description="Summary of what you already get.")

async def search_internet(query: str) -> str:
    """Search internet for more information."""
    return "Nothing found"

async def get_user_input(query: str) -> str:
    """Get additional information from user."""
    return ("The goal is to get the image of the green cells under IF staining and count the number of cells in the image."
    "The software should accept user uploaded files, has a simple web interface, the code should meet PEP8 standard, the rest can be decided by the developer.")

async def clarify(query: GetExtraInformation, role: Role) -> UserClarification:
    """Clarify the requirements."""
    return await role.aask(query, UserClarification)



class FormDialogInfo(BaseModel):
    """Create a JSON Schema Form Dialog using react-jsonschema-form to get more information from the user.
    Whenever possible, try to propose the options for the user to choose from, instead of asking the user to type in the text."""
    form_schema: str = Field(description="json schema for the fields, in yaml format")
    ui_schema: Optional[str] = Field(None, description="customized ui schema for rendering the form, json string, no need to escape quotes, in yaml format")
    submit_label: Optional[str] = Field("Submit", description="Submit button label")


@pytest.mark.asyncio
async def test_schema_user():
    def get_user_response(config: FormDialogInfo) -> UserClarification:
        """Get user response."""
        return UserClarification(content=str({"anwser": "I don't know"}))
    User = Role.create(name="Bob",
        profile="User",
        goal="Provide the use case and requirements for the development, the aim is to create a cell counting software for U2OS cells under confocal microscope, I would like to use web interface in Imjoy.",
        constraints=None,
        actions=[get_user_response])
    user = User()
    # create a form_schema for get user name
    form_schema = json.dumps({"title": "Get User Name", "type": "object", "properties": {"name": {"type": "string"}}})
    form_dialog = FormDialogInfo(form_schema=form_schema)
    msg = Message(content=form_dialog.json(), instruct_content=form_dialog, role="Boss")
    responses = await user.handle(msg)
    assert isinstance(responses[0].instruct_content, UserClarification)

@pytest.mark.asyncio
async def test_schema_str_input():
    async def create_user_requirements(query: str, role: Role) -> SoftwareRequirementDocument:
        """Create user requirements."""
        response = await role.aask(query, Union[SoftwareRequirementDocument, GetExtraInformation])
        if isinstance(response, SoftwareRequirementDocument):
            return response
        elif isinstance(response, GetExtraInformation):
            user_req = await get_user_input(response)
            return await role.aask(user_req, SoftwareRequirementDocument)
        else:
            raise TypeError(f"response must be SoftwareRequirementDocument or GetExtraInformation, but got {type(response)}")

    BioImageAnalyst = Role.create(name="Alice",
                profile="BioImage Analyst",
                goal="Efficiently communicate with the user and translate the user's needs into software requirements",
                constraints=None,
                actions=[create_user_requirements])

    bio = BioImageAnalyst()
    responses = await bio.handle(Message(role="Bot", content="Create a segmentation software"))
    assert isinstance(responses[0].instruct_content, SoftwareRequirementDocument)

@pytest.mark.asyncio
async def test_schemas():
    User = Role.create(name="Bob",
                profile="User",
                goal="Provide the use case and requirements for the development, the aim is to create a cell counting software for U2OS cells under confocal microscope, I would like to use web interface in Imjoy.",
                constraints=None,
                actions=[clarify, search_internet])
    user = User()
    responses = await user.handle(Message(role="Bot", content="Find more information on the internet about cell counting software."))
    assert responses[0].instruct_content is None
    assert responses[0].content == "Nothing found"

    instruct = GetExtraInformation(content="Tell me the use case in 1 sentence.", summary="Requesting details about the use case")
    responses = await user.handle(Message(role="Bot", content=instruct.json(), instruct_content=instruct))
    assert isinstance(responses[0].instruct_content, UserClarification)



@pytest.mark.asyncio
async def test_schema_str():
    async def process_user_input(query: GetExtraInformation, role: Role) -> str:
        """Process user input."""
        query = await role.aask(query, SearchInternetQuery)
        return query

    User = Role.create(name="Bob",
                profile="User",
                goal="Provide the use case and requirements for the development, the aim is to create a cell counting software for U2OS cells under confocal microscope, I would like to use web interface in Imjoy.",
                constraints=None,
                actions=[process_user_input])
    user = User()
    instruct = GetExtraInformation(content="Tell me the use case in 1 sentence.", summary="Requesting details about the use case")
    responses = await user.handle(Message(role="Bot", content=instruct.json(), instruct_content=instruct))
    assert isinstance(responses[0].instruct_content, SearchInternetQuery)