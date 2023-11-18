import asyncio
from functools import partial
from typing import Union
import yaml
from schema_agents.role import Role
from schema_agents.teams import Team

from schema_agents.teams.image_analysis_hub.data_engineer import create_data_engineer
from schema_agents.teams.image_analysis_hub.schemas import (GenerateUserForm, SoftwareRequirement, UserClarification,
                      UserRequirements)
from schema_agents.teams.image_analysis_hub.web_developer import ReactUI, create_web_developer
from schema_agents.teams.image_analysis_hub.microscopist import create_microscopist, MicroscopeControlRequirements

async def clarify_user_request(client, user_query: str, role: Role) -> Union[UserClarification, MicroscopeControlRequirements]:
    """Clarify user request by prompting to the user with a form."""
    config = await role.aask(user_query, Union[GenerateUserForm, MicroscopeControlRequirements])
    if isinstance(config, MicroscopeControlRequirements):
        return config
    fm = await client.createWindow(
        src="https://oeway.github.io/imjoy-json-schema-form/",
        config={
            "schema": config.form_schema and yaml.safe_load(config.form_schema),
            "ui_schema": config.ui_schema and yaml.safe_load(config.ui_schema),
            "submit_label": config.submit_label,
        },
        height="400px"
    )
    form = await fm.get_data()
    return UserClarification(user_query=user_query, form_data=str(form['formData']))

async def create_user_requirements(req: UserClarification, role: Role) -> Union[MicroscopeControlRequirements, UserRequirements]:
    """Respond to user's requests (can be control microscope or create software) after clarification."""
    return await role.aask(req, Union[MicroscopeControlRequirements, UserRequirements])

async def create_software_requirements(req: UserRequirements, role: Role) -> SoftwareRequirement:
    """Create software requirement."""
    return await role.aask(req, SoftwareRequirement)

async def deploy_app(ui: ReactUI, role: Role):
    """Deploy the app for sharing."""
    # serve_plugin(ui)
    print("Deploying the app...")

def create_image_analysis_hub(client, investment):
    """Create a team for image analysis."""
    team = Team(name="Image Analysis Hub", profile="A team of roles to create software for image analysis.", goal="Create a software for image analysis.", investment=investment)
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
    # devops = Role(name="Bruce",
    #             profile="DevOps",
    #             goal="Deploy the software to the cloud and make it available to the user.",
    #             constraints=None,
    #             actions=[deploy_app])  
    
    # microscopist = create_microscopist(client=client)
    
    team.hire([ux_manager, project_manager, web_developer, data_engineer])
    return team
