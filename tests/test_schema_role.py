#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from typing import List, Optional
from pydantic import BaseModel, Field
from schema_agents.role import Role, Action
from metagpt.software_company import SoftwareCompany
from metagpt.schema import Message
import json
from functools import partial

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

@pytest.mark.asyncio
async def test_schema_str():
    User = Role.create(name="Bob",
                profile="User",
                goal="Provide the use case and requirements for the development, the aim is to create a cell counting software for U2OS cells under confocal microscope, I would like to use web interface in Imjoy.",
                constraints=None,
                actions=[search_internet],
                watch=[GetExtraInformation])

    company = SoftwareCompany()
    company.hire([User()])
    company.invest(0.1)
    instruct = GetExtraInformation(content="Tell me the use case in 1 sentence.", summary="Requesting details about the use case")
    company.environment.publish_message(Message(role="Bot", content=instruct.json(), instruct_content=instruct, cause_by=Action.get(GetExtraInformation)))
    assert company.environment.memory.get_by_action(Action.get(GetExtraInformation))
    await company.environment.run()
    assert company.environment.memory.get_by_action(Action.get(search_internet))
    
@pytest.mark.asyncio
async def test_schemas():
    User = Role.create(name="Bob",
                profile="User",
                goal="Provide the use case and requirements for the development, the aim is to create a cell counting software for U2OS cells under confocal microscope, I would like to use web interface in Imjoy.",
                constraints=None,
                actions=[UserClarification, search_internet],
                watch=[GetExtraInformation])

    company = SoftwareCompany()
    company.hire([User()])
    company.invest(0.1)
    instruct = GetExtraInformation(content="Tell me the use case in 1 sentence.", summary="Requesting details about the use case")
    company.environment.publish_message(Message(role="Bot", content=instruct.json(), instruct_content=instruct, cause_by=Action.get(GetExtraInformation)))
    assert company.environment.memory.get_by_action(Action.get(GetExtraInformation))
    await company.environment.run()
    assert company.environment.memory.get_by_action(Action.get(UserClarification))


@pytest.mark.asyncio
async def test_schema_str_input():
    BioImageAnalyst = Role.create(name="Alice",
                profile="BioImage Analyst",
                goal="Efficiently communicate with the user and translate the user's needs into software requirements",
                constraints=None,
                actions=[SoftwareRequirementDocument, get_user_input],
                watch=[UserClarification, get_user_input])

    company = SoftwareCompany()
    company.hire([BioImageAnalyst()])
    company.invest(0.1)
    idea = "Create a segmentation software"
    company.environment.publish_message(Message(role="Bot", content=idea, cause_by=Action.get(UserClarification)))
    assert company.environment.memory.get_by_action(Action.get(UserClarification))
    await company.environment.run()
    assert company.environment.memory.get_by_action(Action.get(SoftwareRequirementDocument))
    await company.environment.run()


@pytest.mark.asyncio
async def test_schema_role():
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

    company = SoftwareCompany()
    # company.hire([ProductManager(), Architect(), ProjectManager(), Engineer(n_borg=5)])
    company.hire([BioImageAnalyst(), User()])
    company.invest(0.1)
    idea = "How can I help you? Please provide a short anwser in 1 sentence."
    company.environment.publish_message(Message(role="Bot", content=idea, cause_by=Action.get(GetExtraInformation)))
    assert company.environment.memory.get_by_action(Action.get(GetExtraInformation))
    await company.environment.run()
    assert company.environment.memory.get_by_action(Action.get(UserClarification))


class FormDialogInfo(BaseModel):
    """Create a JSON Schema Form Dialog using react-jsonschema-form to get more information from the user.
    Whenever possible, try to propose the options for the user to choose from, instead of asking the user to type in the text."""
    form_schema: str = Field(description="json schema for the fields, in yaml format")
    ui_schema: Optional[str] = Field(None, description="customized ui schema for rendering the form, json string, no need to escape quotes, in yaml format")
    submit_label: Optional[str] = Field("Submit", description="Submit button label")


@pytest.mark.asyncio
async def test_schema_user():
    def get_user_response(client, config: FormDialogInfo) -> UserClarification:
        return UserClarification(form_data=str({"anwser": "I don't know"}))
    GetUserResponse = Action.create(partial(get_user_response, {}))
    FormDialogInfoAction = Action.create(FormDialogInfo)
    User = Role.create(name="Bob",
        profile="User",
        goal="Provide the use case and requirements for the development, the aim is to create a cell counting software for U2OS cells under confocal microscope, I would like to use web interface in Imjoy.",
        constraints=None,
        actions=[GetUserResponse],
        watch=[FormDialogInfoAction])
    user = User()
    # create a form_schema for get user name
    form_schema = json.dumps({"title": "Get User Name", "type": "object", "properties": {"name": {"type": "string"}}})
    form_dialog = FormDialogInfo(form_schema=form_schema)
    msg = Message(content=form_dialog.json(), instruct_content=form_dialog, role="Boss", cause_by=FormDialogInfoAction)
    user.recv(msg)
    assert msg in user._rc.important_memory
    await user._react()
    