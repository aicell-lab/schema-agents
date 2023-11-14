from typing import List, Optional

from pydantic import BaseModel, Field
from enum import Enum

# Common description for Python function scripts
common_desc = {
    "script": "Python script that defines functions according to the requirements. Includes imports, function definition, logic, and implementation.",
    "names": "List of function names.",
    "pip_packages": "Required Python pip packages. Reuse existing libraries and prioritize common libraries.",
    "test_script": "Script for testing the Python function. Includes test cases, validation logic, and assertions.",
    "docstring": "Brief notes for usage, debugging, potential error fixing, and further improvements."
}


class GenerateUserForm(BaseModel):
    """Based on user query, create a JSON Schema Form Dialog using react-jsonschema-form to get more information about what the user want to create.
    The aim is to gather the information needed to create the software/tool requirement document or perform data acquisition.
    Whenever possible, try to propose the options for the user to choose from, instead of asking the user to type in the text."""
    form_schema: str = Field(description="json schema for the fields, in yaml format; MUST be a valid YAML file")
    ui_schema: Optional[str] = Field(None, description="customized ui schema for rendering the form, json string, no need to escape quotes, in yaml format; MUST be a valid YAML file")
    submit_label: Optional[str] = Field("Submit", description="Submit button label")

class UserRequirements(BaseModel):
    """User requirements for developing a software."""
    goal: str = Field(description="The goal of the user.")
    ui: str = Field(description="Requirements for the user interface.")
    data: str = Field(description="Requirements for the data.")
    validation: str = Field(description="Additional information for testing, e.g. test data and the expected outcome")
    notes: str = Field(description="Additional notes.")

class UserClarification(BaseModel):
    """User submitted form data."""
    user_query: str = Field(description="The original user query")
    form_data: str = Field(description="Form data in json format")

class GetExtraInformation(BaseModel):
    """Extra information needed to be able to work on the task."""
    content: str = Field(description="The information.")
    summary: str = Field(description="Summary of what you already get.")


class FunctionMemory(BaseModel):
    """Functions to be saved in the long term memory."""
    function_name: str = Field(default="", description="Function name")
    code: str = Field(default="", description="original code of the function")
    lang: str = Field(default="", description="function language")
    args: List[str] = Field(default=[], description="arguments of the function")
    
class ErrorMemory(BaseModel):
    """Experience of making errors to be saved in the long term memory."""    
    error: str = Field(default="", description="Error description")
    cause_by: str = Field(default="", description="Cause of the error")
    solution: str = Field(default="", description="Solution to fix the error")

class ExperienceMemory(BaseModel):
    """Experience to be saved in the long term memory."""    
    summary: str = Field(default="", description="Summary of the experience. This will be used as index in the long term memory for retrival.")
    keypoints: Optional[str] = Field(default="", description="Key points to remember for this experience.")

CODING_RULES = """
Important Rules for Coding:
- Use `window.python` to refer the external python functions
- Use tailwindcss for styling (the page has `https://cdn.tailwindcss.com` loaded)
- DO NOT user other libraries besides React and React DOM
"""

class ReactUI(BaseModel):
    """Defines the ImJoy UI plugin using React."""
    id: str = Field(..., description="a short id of the application")
    root_element: str = Field("root", description="The root element ID where the React app will be attached.")
    react_version: str = Field("17.0.2", description="Version of React to use.")
    react_dom_version: str = Field("17.0.2", description="Version of ReactDOM to use.")
    babel_version: str = Field("6.26.0", description="Version of Babel to transpile JSX.")
    jsx_script: str = Field(..., description="JSX script defining the React app. It will be transpiled with Babel and the dom should be mounted to the root element. DO NOT use import statements." + CODING_RULES) # In the script it must export a set of function as the ImJoy plugin API. e.g.: `api.export({setup, run, show_image: showImage, etc....})
    # test_script: str = Field(..., description="Test script for calling the exported ImJoy API for testing. Should include test cases, expected outcomes, and validation logic.")

class PythonFunctionScript(BaseModel):
    """Represents a Python function and test script with all its properties."""
    function_names: List[str] = Field(..., description=common_desc['names'])
    function_script: str = Field(..., description=common_desc['script'])
    pip_packages: List[str] = Field(..., description=common_desc['pip_packages'])
    test_script: str = Field(..., description=common_desc['test_script'])
    docstring: Optional[str] = Field(None, description=common_desc['docstring'])
   

class PythonFunctionScriptWithLineNumber(BaseModel):
    """Python function and test script with line number for display and editing"""
    function_script_with_line_number: str = Field(..., description=common_desc['script'])
    pip_packages: List[str] = Field(..., description=common_desc['pip_packages'])
    test_script_with_line_number: str = Field(..., description=common_desc['test_script'])
    docstring: Optional[str] = Field(None, description=common_desc['docstring'])

# Enum for operations
class OperationEnum(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


# Description for Change operations
change_desc = {
    "operation": "The operation type ('add', 'remove', 'replace')",
    "line": "The line number (of the original script) affected by the change",
    "content": "The content to be added or replaced"
}

# BaseModel for changes in Python function scripts
class Change(BaseModel):
    operation: OperationEnum = Field(..., description=change_desc['operation'])
    line: int = Field(..., description=change_desc['line'])
    content: str = Field(None, description=change_desc['content'])

class PythonFunctionScriptChanges(BaseModel):
    """Changes to the Python function and test script."""
    explanation: str = Field(..., description="Brief explanation on why the changes is needed.")
    function_script_changes: Optional[List[Change]] = Field(None, description="List of changes to the function script.")
    updated_pip_packages: List[str] = Field(..., description="Updated pip packages.")
    test_script_changes: Optional[List[Change]] = Field(None, description="List of changes to the test script.")
    experience: Optional[ExperienceMemory] = Field(None, description="Experience for successfully fixing the errors, this will be noted down in the long term memory to avoid making the same mistake again.")

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
    python_function_requirements: Optional[list[PythonFunctionRequirement]] = Field(..., description="A list of requirements for the python functions which will be called in the web UI.")
    react_ui_requirements: Optional[ReactUIRequirement] = Field(description="User interface requirements for the react.js web UI. The UI will be used to interact with the user and call the python functions (made available under a global variable named `pythonFunctions` using imjoy-rpc). E.g. `pythonFunctions.load_data(...)` can be called in a button click callback.")
    additional_notes: str = Field(default="", description="Any additional notes or requirements that need to be considered.")


