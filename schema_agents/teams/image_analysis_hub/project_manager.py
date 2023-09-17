from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import asyncio
from schema_agents.role import Role, Action
from schema_agents.schema import Message
from .user import UserRequirement, UserRequirement
class PythonFunctionRequirement(BaseModel):
    """Python Function Requirement
    Providing detailed information for implementing the python function."""
    function_name: str = Field(..., description="Name of the python function to be implemented.")
    function_signature: str = Field(..., description="Signature of the python function, outlining the expected parameters and return type.")
    feature_instructions: str = Field(..., description="Clear instructions on how to implement the feature, including code snippets, logic, and algorithms if necessary.")
    context_requirements: Optional[str] = Field(default=None, description="Additional context or prerequisites required for the python function, including dependencies on other functions, modules, or data.")
    testing_requirements: Optional[str] = Field(default=None, description="Instructions or guidelines on how to test the python function, including expected outcomes, edge cases, and validation criteria.")


class ReactUIRequirement(BaseModel):
    """React UI Requirement
    The aim is to create a web UI for the main script in python to obtain user input, display results, and allow for interaction. 
    The exported imjoy plugin api function will be called inside the main function to interact with the user."""
    plugin_name: str = Field(..., description="Name of the React plugin for main function referencing.")
    ui_layout: str = Field(..., description="Description of the UI layout, including positioning of elements.")
    interaction_patterns: str = Field(..., description="Details of interaction patterns, such as clicks, swipes, etc.")
    functionalities: str = Field(..., description="Details of the functions that need to be implemented, such as selecting an image, segmenting cells, etc.")
    user_flows: str = Field(..., description="Outline of the user flow, describing how the user moves through the application.")
    # imjoy_plugin_api: List[ReactUIApiFunction] = Field(..., description="List of functions for the main function to call. These functions are used to configure the UI, display results, register callbacks, etc.")

class SoftwareRequirement(BaseModel):
    """Software Requirement
    The the software requirement is to used to instruct the developers to create a set of python functions with web UI built with react.js according to user's request.
    """
    id: str = Field(..., description="a short id of the application")
    original_requirements: str = Field(..., description="The polished complete original requirements from the user.")
    python_function_requirements: Optional[List[PythonFunctionRequirement]] = Field(..., description="A list of requirements for the python functions which will be called in the web UI.")
    react_ui_requirements: Optional[ReactUIRequirement] = Field(description="User interface requirements for the react.js web UI. The UI will be used to interact with the user and call the python functions (made available under a global variable named `pythonFunctions` using imjoy-rpc). E.g. `pythonFunctions.load_data(...)` can be called in a button click callback.")
    additional_notes: str = Field(default="", description="Any additional notes or requirements that need to be considered.")

SoftwareRequirementAction = Action.create(SoftwareRequirement)
ProjectManager = Role.create(name="Bob",
                                profile="Project Manager",
                                goal="Translate user requirements into a detailed software requirement specification, create a python function script which solves user's request.",
                                constraints=None,
                                actions=[SoftwareRequirementAction])

async def main():
    user_requirement = UserRequirement(
        task_description="Load an image file and segment the cells in the image, count the cells then show the result image.",
        input_data_type="image",
        desired_output="the number of segmented cells in the image",
        test_input_data="use the image at /Users/wei.ouyang/workspace/LibreChat/chatbot/tests/data/img16.png",
        test_desired_output="the number of cells in the image should be more than 12",
        # preferred_interaction="File selection dialog for image input, button to start segmentation",
        additional_notes="the cells are U2OS cells in a IF microscopy image, cells are round and in green color, the background is black."
    )
        
    prod = ProjectManager()

    msg = Message(content=user_requirement.json(), instruct_content=user_requirement, role="User", cause_by=UserRequirementAction)
    prod.recv(msg)
    req = await prod._react()
    print(req)

if __name__ == "__main__":
    asyncio.run(main())