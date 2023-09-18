import asyncio
from functools import partial

import yaml
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from schema_agents.tools.code_interpreter import create_mock_client

from .data_engineer import create_data_engineer
from .schemas import (GenerateUserForm, SoftwareRequirement, UserClarification,
                      UserRequirements)
from .web_developer import ReactUI, create_web_developer


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

class ImageAnalysisHub(Team):
    """
    ImageAnalysisHub: a team of roles to create software for image analysis.
    """
    def recruit(self, client):
        """recruit roles to cooperate"""
        UXManager = Role.create(name="Luisa",
            profile="UX Manager",
            goal="Focus on understanding the user's needs and experience. Understand the user needs by interacting with user and communicate these findings to the project manager by calling `UserRequirements`.",
            constraints=None,
            actions=[partial(clarify_user_request, client), create_user_requirements])

        ProjectManager = Role.create(name="Alice",
                    profile="Project Manager",
                    goal="Efficiently communicate with the user and translate the user's needs into software requirements",
                    constraints=None,
                    actions=[create_software_requirements])

        WebDeveloper  = create_web_developer(client=client)
        DataEngineer = create_data_engineer(client=client)
        DevOps = Role.create(name="Bruce",
                    profile="DevOps",
                    goal="Deploy the software to the cloud and make it available to the user.",
                    constraints=None,
                    actions=[deploy_app])  
        self.environment.add_roles([UXManager(), ProjectManager(), DataEngineer(), WebDeveloper(), DevOps()]) 


async def test():
    lab = ImageAnalysisHub()
    lab.invest(0.5)
    lab.recruit(client=create_mock_client())
    lab.start("create a cell counting software")
    await lab.run(n_round=10)

if __name__ == "__main__":
    asyncio.run(test())