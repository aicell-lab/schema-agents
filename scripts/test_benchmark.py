import asyncio
import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.teams import Team
from schema_agents.schema import Message
from schema_agents.tools.code_interpreter import create_mock_client
from schema_agents.utils.common import EventBus
from schema_agents.logs import logger

# Common description for Python function scripts
common_desc = {
    "script": "Python script that defines functions according to the requirements. Includes imports, function definition, logic, and implementation.",
    "names": "List of function names.",
    "pip_packages": "Required Python pip packages. Reuse existing libraries and prioritize common libraries.",
    "test_script": "Script for testing the Python function. Includes test cases, validation logic, and assertions.",
    "docstring": "Brief notes for usage, debugging, potential error fixing, and further improvements."
}

INSTALL_SCRIPT = """
try:
    import pyodide
    import micropip
    await micropip.install([{packages}])
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', {packages}])
"""


DEPLOY_SCRIPT = """
import asyncio
from imjoy_rpc.hypha import connect_to_server

{function_code}

if not 'hypha_server' in vars() or 'hypha_server' in globals():
    hypha_server = await connect_to_server(
        {{"name": "test client", "server_url": "https://ai.imjoy.io/", "method_timeout": 3000}}
    )

service_config = {{
    "name": "Hypha",
    "id": "{service_id}",
    "config": {{
        "visibility": "public"
    }},
}}

for function_name in {function_names}:
    service_config[function_name] = globals()[function_name]

await hypha_server.register_service(service_config)
"""

UX_MANAGER_PROMPT = """
Generate a detailed UserRequirements based on the {user_query} related to software requirements. The UserRequirements must contain the following information:

- goal: The goal of the user.
- data: Requirements for the data.
- validation: Additional information for testing, e.g., test data and the expected outcome.
- notes: Additional notes.

Ensure that the response is clear, informative, and covers all aspects of the user's inquiry.
"""

PROJECT_MANAGER_PROMPT = """
After receiving detailed UserRequirements from the UX manager regarding software needs, your task is to compile a comprehensive SoftwareRequirement. This Pydantic class includes the following fields:

id: A concise identifier for the application.
original_requirements: The refined and complete set of original requirements from the user.
python_function_requirements: A list specifying the requirements for the Python functions to be executed. Ensure these are represented as PythonFunctionRequirement objects.
additional_notes: Any supplementary notes or requirements that demand consideration.

Your response should meticulously incorporate all facets mentioned in the UserRequirements, offering a clear and detailed SoftwareRequirement. Pay particular attention to accurately translating the Python function requirements into a well-structured list of PythonFunctionRequirement objects.
"""

DATA_ENGINEER_PROMPT = """
"""

class PythonFunctionRequirement(BaseModel):
    """Python Function Requirement
    Providing detailed information for implementing the python function."""
    function_name: str = Field(..., description="Name of the python function to be implemented.")
    function_signature: str = Field(..., description="Signature of the python function, outlining the expected parameters and return type.")
    feature_instructions: str = Field(..., description="Clear instructions on how to implement the feature, including code snippets, logic, and algorithms if necessary.")
    context_requirements: Optional[str] = Field(default=None, description="Additional context or prerequisites required for the python function, including dependencies on other functions, modules, or data.")
    testing_requirements: Optional[str] = Field(default=None, description="Instructions or guidelines on how to test the python function, including expected outcomes, edge cases, and validation criteria.")


class PythonFunctionScript(BaseModel):
    """Represents a Python function and test script with all its properties."""
    function_names: List[str] = Field(..., description=common_desc['names'])
    function_script: str = Field(..., description=common_desc['script'])
    pip_packages: List[str] = Field(..., description=common_desc['pip_packages'])
    test_script: str = Field(..., description=common_desc['test_script'])
    docstring: Optional[str] = Field(None, description=common_desc['docstring'])

class UserRequirements(BaseModel):
    """User requirements for the software."""
    goal: str = Field(description="The goal of the user.")
    data: str = Field(description="Requirements for the data.")
    validation: str = Field(description="Additional information for testing, e.g. test data and the expected outcome")
    notes: str = Field(description="Additional notes.")

class SoftwareRequirement(BaseModel):
    """Software Requirement
    The the software requirement is to used to instruct the developers to create a set of python functions according to user's request.
    """
    id: str = Field(..., description="a short id of the application")
    original_requirements: str = Field(..., description="The polished complete original requirements from the user.")
    python_function_requirements: Optional[list[PythonFunctionRequirement]] = Field(..., description="A list of requirements for the python functions which will be executed.")
    additional_notes: str = Field(default="", description="Any additional notes or requirements that need to be considered.")



    
async def schema_create_user_requirements(req: str, role: Role) -> UserRequirements:
    """Create user requirement."""
    return await role.aask(req, UserRequirements)

async def create_user_requirements(req: str, role: Role) -> str:
    """Create user requirement."""
    return await role.aask(req, str)


async def schema_create_software_requirements(req: UserRequirements, role: Role) -> SoftwareRequirement:
    """Create software requirement."""
    return await role.aask(req, SoftwareRequirement)

async def create_software_requirements(req: str, role: Role) -> str:
    """Create software requirement."""
    return await role.aask(req, str)

async def schema_develop_python_functions(req: SoftwareRequirement, role: Role) -> PythonFunctionScript:
        """Develop python functions based on software requirements."""
        async def generate_code(req: SoftwareRequirement, role: Role) -> PythonFunctionScript:
            return await role.aask(req, PythonFunctionScript)

        
        async def test_run_python_function(role, client, service_id, python_function: PythonFunctionScript) -> PythonFunctionScript:
            """Test run the python function script."""
            if python_function.pip_packages:
                packages = ",".join([f"'{p}'" for p in python_function.pip_packages])
                results = await client.executeScript({"script": INSTALL_SCRIPT.format(packages=packages)})
                output_summary = json.dumps(
                    {k: results[k] for k in results.keys() if results[k]}, indent=1
                )
                if results["status"] != "ok":
                    raise RuntimeError(f"Failed to install pip packages: {python_function.pip_packages}, error: {output_summary}")
            results = await client.executeScript(
                {"script": python_function.function_script + "\n" + python_function.test_script}
            )
            
            # deploy the functions
            results = await client.executeScript(
                {"script": DEPLOY_SCRIPT.format(function_names=python_function.function_names,
                                    service_id=service_id,
                                    function_code=python_function.function_script
                                    )
                }
            )
            return python_function
        client = create_mock_client()
        # if isinstance(req, SoftwareRequirement):
        func = await generate_code(req, role)
        try:
            func = await test_run_python_function(role, client, req.id, func)
        except RuntimeError as exp:
            req.additional_notes += f"\nPlease avoid the following error: {exp}"
            func = await generate_code(req, role)
            func = await test_run_python_function(role, client, req.id, func)
        return func
    
    
    
    
def create_schema_team(investment):
    """Create a team of schema agents."""
    schema_team = Team(name="Schema agents team", profile="A team of schema agents for different roles.", goal="Work as team to implement user's query.", investment=investment)
    ux_manager = Role(name="Luisa",
            profile="UX Manager",
            goal="Focus on understanding the user's needs and experience. Understand the user needs and communicate these findings to the project manager by calling `UserRequirements`.",
            constraints=None,
            actions=[schema_create_user_requirements])

    project_manager = Role(name="Alice",
                    profile="Project Manager",
                    goal="Efficiently communicate with ux manager and translate the user's needs `UserRequirements` into software requirements `SoftwareRequirement`.",
                    constraints=None,
                    actions=[schema_create_software_requirements])

    data_engineer = Role(
            name="Alice",
            profile="Data Engineer",
            goal="Develop the python function script according to the software requirement `SoftwareRequirement`, ensuring that it fulfills the desired functionality. Implement necessary algorithms, handle data processing, and write tests to validate the correctness of the function.",
            constraints=None,
            actions=[schema_develop_python_functions],
        )
    schema_team.hire([ux_manager, project_manager, data_engineer])
    return schema_team

