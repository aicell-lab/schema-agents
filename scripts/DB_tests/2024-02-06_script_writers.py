import os
import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
from xml.etree import ElementTree as ET



class Requirements(BaseModel):
    """The requirements for the project"""
    development_plan: str = Field(..., description="The overall plan for implementing the user's request")
    required_components: str = Field(..., description="The components required to implement the user's request")
    external_dependencies : str = Field(..., description="The external dependencies required to implement the user's request")

class Function(BaseModel):
    """A function in the project"""
    name: str = Field(..., description="The name of the function")
    description: str = Field(..., description="A detailed description of the function")
    inputs: str = Field(..., description="The inputs of the function")
    outputs: str = Field(..., description="The outputs of the function")

class FileSpec(BaseModel):
    """A file in the project"""
    name: str = Field(..., description="The name of the file")
    description: str = Field(..., description="A detailed description of the file")
    functions: List[Function] = Field(..., description="The functions the file should contain")

class CodeArchitecture(BaseModel):
    """The code architecture for the project"""
    files : List[FileSpec] = Field(..., description="A list of files in the project and which functions they should contain")

async def make_requirements(user_input : str, role : Role = None) -> Requirements:
    """Sets the requirements for the project including development, testing, and deployment requirements"""
    result = await role.aask(user_input, Requirements)
    return(result)

async def make_architecture(requirements : Requirements, role : Role = None) -> CodeArchitecture:
    """Sets the code architecture for the project"""
    result = await role.aask(requirements, CodeArchitecture)
    return(result)

class Script(BaseModel):
    """A fully-implemented script. The script MUST be completely filled out, it should not be missing any code."""
    filename: str = Field(..., description="The filename of the script. It should contain no spaces (only underscores)")
    contents: str = Field(..., description="The contents of the script")
    requirements : list[str] = Field(..., description="The requirements for the script, each element is a line in a requirements.txt file")

class ProjectScripts(BaseModel):
    """A list of the project's scripts"""
    scripts: List[Script] = Field(..., description="A list of project scripts")

async def write_scripts(architecture : CodeArchitecture, role : Role = None) -> ProjectScripts:
    """Writes the scripts for the project"""
    dependencies = []
    project_scripts = []
    for file_spec in architecture.files:
        script = await role.aask(file_spec, Script)
        dependencies.extend(script.requirements)
        output_dir = "generated_scripts"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, script.filename), "w") as file:
            file.write(script.contents)
        project_scripts.append(script)
    with open(os.path.join(output_dir, "requirements.txt"), "w") as file:
        file.write("\n".join(list(set(dependencies))))
    return(ProjectScripts(scripts=project_scripts))


def make_team():
    agents = []
    project_manager = Role(
        name="Project Manager",
        profile="The project manager that oversees the entire project",
        goal="To set goals for the project and ensure that the project is on track",
        constraints=None,
        actions=[make_requirements],
    )
    agents.append(project_manager)

    code_architect = Role(
        name="Code Architect",
        profile="The code architect that sets the code architecture for the project",
        goal="To set the code architecture for the project",
        constraints=None,
        actions=[make_architecture],
    )
    agents.append(code_architect)

    software_engineer = Role(
        name="Software Engineer",
        profile="A software engineer that writes the scripts for the project",
        goal="To write the scripts for the project",
        constraints=None,
        actions=[write_scripts],
    )
    agents.append(software_engineer)

    team = Team(name="Software engineering team", profile="A team of agents meant to write complete, portable, usable, and error-free software packages", investment=0.7)

    team.hire(agents)
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
    return(team)

# Main function
async def main():
    team = make_team()
    user_request = "I want a software package that will identify differentially expressed genes between sample CLM 01 and CLM 02 using the following workflow: 'Load gene expression count data from GSM8041061_CLM_01_matrix.mtx.gz and GSM8041062_CLM_02_matrix.mtx.gz', 'Normalize the data using scale factors from GSM8041061_CLM_01_scalefactors_json.json.gz and GSM8041062_CLM_02_scalefactors_json.json.gz', 'Perform differential expression analysis using statistical methods such as DESeq2 or edgeR', 'Visualize the results with heatmaps or volcano plots'"
    responses = await team.handle(Message(content=user_request, role="User"))
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()