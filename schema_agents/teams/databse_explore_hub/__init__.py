
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
from functools import partial

from schema_agents.schema import Message
from schema_agents.role import Role
from .project_manager import SoftwareRequirement
from .data_engineer import create_data_engineer
from .web_developer import create_web_developer, ReactUI
from schema_agents.tools.code_interpreter import create_mock_client
from schema_agents.teams import Team

import yaml

class GenerateUserForm(BaseModel):
    """Based on user query, create a JSON Schema Form Dialog using react-jsonschema-form to get more information about what the user want to create.
    The aim is to gather the information needed to create the software requirement document.
    Whenever possible, try to propose the options for the user to choose from, instead of asking the user to type in the text."""
    form_schema: str = Field(description="json schema for the fields, in yaml format")
    ui_schema: Optional[str] = Field(None, description="customized ui schema for rendering the form, json string, no need to escape quotes, in yaml format")
    submit_label: Optional[str] = Field("Submit", description="Submit button label")

class UserRequirements(BaseModel):
    """User requirements for the software."""
    goal: str = Field(description="The goal of the user.")
    ui: str = Field(description="Requirements for the user interface.")
    data: str = Field(description="Requirements for the data.")
    validation: str = Field(description="Additional information for testing, e.g. test data and the expected outcome")
    notes: str = Field(description="Additional notes.")
    
class UserClarification(BaseModel):
    """User submitted form data."""
    form_data: str = Field(description="Form data in json format")

class GetExtraInformation(BaseModel):
    """Extra information needed to be able to work on the task."""
    content: str = Field(description="The information.")
    summary: str = Field(description="Summary of what you already get.")

async def show_message(client, message: Message):
    """Show message to the user."""
    await client.set_output(message.content)

async def clarify_user_request(client, user_query: str, role: Role) -> UserClarification:
    """Clarify user request by prompting to the user with a form."""
    config = await role.aask(user_query, GenerateUserForm)
    fm = await client.show_dialog(
        src="https://oeway.github.io/imjoy-json-schema-form/",
        config={
            "schema": config.form_schema and yaml.safe_load(config.form_schema),
            "ui_schema": config.ui_schema and yaml.safe_load(config.ui_schema),
            "submit_label": config.submit_label,
        }
    )
    form = await fm.get_data()
    return UserClarification(form_data=str(form['formData']))

async def create_user_requirements(req: UserClarification, role: Role) -> UserRequirements:
    """Create user requirement."""
    return await role.aask(req, UserRequirements)

async def create_software_requirements(req: UserRequirements, role: Role) -> SoftwareRequirement:
    """Create software requirement."""
    return await role.aask(req, SoftwareRequirement)

async def deploy_app(ui: ReactUI, role: Role):
    """Deploy the app for sharing."""
    # serve_plugin(ui)
    print("Deploying the app...")


def create_hpa_database_explorer(client, investment):
    """recruit roles to cooperate"""
    team = Team(name="HPA Database Explorer", profile="A team of roles to create software for image analysis.", goal="Create a software for image analysis.", investment=investment)
    ux_manager = Role(name="Luisa",
        profile="UX Manager",
        goal="Focus on understanding the user's needs and experience. Understand the user needs by interacting with user and communicate these findings to the project manager by calling `UserRequirements`.",
        constraints=None,
        actions=[partial(clarify_user_request, client), create_user_requirements])

    project_manager = Role(name="Alice",
                profile="Project Manager",
                goal="Efficiently communicate with the user and translate the user's needs into software requirements",
                constraints=None,
                actions=[create_software_requirements])

    web_developer  = create_web_developer(client=client)
    data_engineer = create_data_engineer(client=client)
    devops = Role(name="Bruce",
                profile="DevOps",
                goal="Deploy the software to the cloud and make it available to the user.",
                constraints=None,
                actions=[deploy_app])  

    team.hire([ux_manager, project_manager, web_developer, data_engineer, web_developer, devops])
    return team

async def test():
    lab = create_hpa_database_explorer(client=create_mock_client())
    await lab.handle("create a cell counting software")

if __name__ == "__main__":
    asyncio.run(test())