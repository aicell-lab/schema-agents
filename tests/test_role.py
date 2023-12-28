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

class UserInput(BaseModel):
    """User clarification"""
    goal: str = Field(description="Clarified goal")
    notes: str = Field(description="Additional notes")

async def get_user_input(query: str) -> str:
    """Get additional information from user."""
    return UserInput(
        goal="The goal is to get the image of the green cells under IF staining and count the number of cells in the image.",
        notes="The software should accept user uploaded files, has a simple web interface, the code should meet PEP8 standard, the rest can be decided by the developer."
    )

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
    user = Role(name="Bob",
        profile="User",
        goal="Provide the use case and requirements for the development, the aim is to create a cell counting software for U2OS cells under confocal microscope, I would like to use web interface in Imjoy.",
        constraints=None,
        actions=[get_user_response])
    user.get_event_bus().register_default_events()
    # create a form_schema for get user name
    form_schema = json.dumps({"title": "Get User Name", "type": "object", "properties": {"name": {"type": "string"}}})
    form_dialog = FormDialogInfo(form_schema=form_schema)
    msg = Message(content=form_dialog.json(), data=form_dialog, role="Boss")
    responses = await user.handle(msg)
    assert isinstance(responses[0].data, UserClarification)

@pytest.mark.asyncio
async def test_str_streaming():
    async def create_user_requirements(query: str, role: Role) -> str:
        """Create user requirements."""
        return await role.aask(query, str)

    bioimage_analyst = Role(name="Alice",
                profile="BioImage Analyst",
                goal="Efficiently communicate with the user and translate the user's needs into software requirements",
                constraints=None,
                actions=[create_user_requirements])
    bioimage_analyst.get_event_bus().register_default_events()
    responses = await bioimage_analyst.handle(Message(role="Bot", content="What can you do for me?"))
    assert isinstance(responses[0].content, str)


@pytest.mark.asyncio
async def test_tool_call():
    async def create_user_requirements(query: str, role: Role) -> SoftwareRequirementDocument:
        """Create user requirements."""
        def create_sdr(sdr: SoftwareRequirementDocument) -> str:
            """Create Software Requirement Document."""
            return "SDR"

        async def get_extra_information(extra_info: GetExtraInformation, hint: str) -> UserInput:
            """Get Extra Information with hint to the user."""
            return await get_user_input(extra_info)

        response = await role.acall(query, [create_sdr, get_extra_information], SoftwareRequirementDocument)
        return response
        
    bioimage_analyst = Role(name="Alice",
                profile="BioImage Analyst",
                goal="Efficiently communicate with the user and translate the user's needs into software requirements",
                constraints=None,
                actions=[create_user_requirements])
    bioimage_analyst.get_event_bus().register_default_events()
    responses = await bioimage_analyst.handle(Message(role="Bot", content="Create a segmentation software"))
    assert isinstance(responses[0].data, SoftwareRequirementDocument)


@pytest.mark.asyncio
async def test_parallel_function_call():
    async def create_user_requirements(query: str, role: Role) -> SoftwareRequirementDocument:
        """Create user requirements."""
        response = await role.aask(query, Union[SoftwareRequirementDocument, GetExtraInformation], use_tool_calls=True)
        if isinstance(response, SoftwareRequirementDocument):
            return response
        elif isinstance(response, GetExtraInformation):
            user_req = await get_user_input(response)
            return await role.aask(user_req, SoftwareRequirementDocument)
        else:
            raise TypeError(f"response must be SoftwareRequirementDocument or GetExtraInformation, but got {type(response)}")

    bioimage_analyst = Role(name="Alice",
                profile="BioImage Analyst",
                goal="Efficiently communicate with the user and translate the user's needs into software requirements",
                constraints=None,
                actions=[create_user_requirements])
    bioimage_analyst.get_event_bus().register_default_events()
    responses = await bioimage_analyst.handle(Message(role="Bot", content="Create a segmentation software"))
    assert isinstance(responses[0].data, SoftwareRequirementDocument)

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

    bioimage_analyst = Role(name="Alice",
                profile="BioImage Analyst",
                goal="Efficiently communicate with the user and translate the user's needs into software requirements",
                constraints=None,
                actions=[create_user_requirements])
    bioimage_analyst.get_event_bus().register_default_events()
    responses = await bioimage_analyst.handle(Message(role="Bot", content="Create a segmentation software"))
    assert isinstance(responses[0].data, SoftwareRequirementDocument)

@pytest.mark.asyncio
async def test_schemas():
    user = Role(name="Bob",
                profile="User",
                goal="Provide the use case and requirements for the development, the aim is to create a cell counting software for U2OS cells under confocal microscope, I would like to use web interface in Imjoy.",
                constraints=None,
                actions=[clarify, search_internet])
    user.get_event_bus().register_default_events()
    responses = await user.handle(Message(role="Bot", content="Find more information on the internet about cell counting software."))
    assert responses[0].data is None
    assert responses[0].content == "Nothing found"

    instruct = GetExtraInformation(content="Tell me the use case in 1 sentence.", summary="Requesting details about the use case")
    responses = await user.handle(Message(role="Bot", content=instruct.json(), data=instruct))
    assert isinstance(responses[0].data, UserClarification)



@pytest.mark.asyncio
async def test_schema_str():
    async def process_user_input(query: GetExtraInformation, role: Role) -> str:
        """Process user input."""
        query = await role.aask(query, SearchInternetQuery)
        return query

    user = Role(name="Bob",
                profile="User",
                goal="Provide the use case and requirements for the development, the aim is to create a cell counting software for U2OS cells under confocal microscope, I would like to use web interface in Imjoy.",
                constraints=None,
                actions=[process_user_input])
    user.get_event_bus().register_default_events()
    instruct = GetExtraInformation(content="Tell me the use case in 1 sentence.", summary="Requesting details about the use case")
    responses = await user.handle(Message(role="Bot", content=instruct.json(), data=instruct))
    assert isinstance(responses[0].data, SearchInternetQuery)


@pytest.mark.asyncio
async def test_respond_user_str():
    async def respond_to_user(query: str, role: Role) -> str:
        """Respond to user."""
        response = await role.aask(query, str)
        return response
        
    role = Role(name="Alice",
                profile="Customer service",
                goal="Efficiently communicate with the user and translate the user's needs to technical requirements",
                constraints=None,
                actions=[respond_to_user])
    responses = await role.handle("Say hello")
    assert responses