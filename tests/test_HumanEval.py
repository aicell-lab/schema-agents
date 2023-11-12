from schema_agents.role import Role
import random
import os
import json
from typing import List, Union, Optional, Dict, Any
from pydantic import BaseModel, Field
from human_eval.data import write_jsonl, read_problems
from schema_agents.teams.image_analysis_hub.schemas import PythonFunctionScript


class PythonOutput(BaseModel):
    """Represents a Python function with all its properties."""
    function_names: List[str] = Field(..., description="Function names in the script.")
    function_script: str = Field(..., description="Completed python function script. This script should be able to pass the test cases. Includes imports, function definition, logic, and implementation.")
    docstring: Optional[str] = Field(None, description="Brief notes for usage, debugging, potential error fixing, and further improvements.")
   

async def generate_code(
    req, role: Role
) -> PythonOutput:
    # retrieve req-related memories from long term memory
    response = await role.aask(req, PythonOutput)
    return response

# async def test_run_python_function(
#     role, client, service_id, python_function: PythonFunctionScript
# ) -> PythonFunctionScript:
#     """Test run the python function script."""
#     if python_function.pip_packages:
#         packages = ",".join([f"'{p}'" for p in python_function.pip_packages])
#         results = await client.executeScript({"script": INSTALL_SCRIPT.format(packages=packages)})
#         output_summary = json.dumps(
#             {k: results[k] for k in results.keys() if results[k]}, indent=1
#         )
#         if results["status"] != "ok":
#             raise RuntimeError(f"Failed to install pip packages: {python_function.pip_packages}, error: {output_summary}")
#     results = await client.executeScript(
#         {"script": python_function.function_script + "\n" + python_function.test_script}
#     )
    # if results["status"] != "ok":
    #     output_summary = json.dumps(
    #         {k: results[k] for k in results.keys() if results[k]}, indent=1
    #     )
        # python_function = await fix_code(role, client, python_function, output_summary)
        # return await test_run_python_function(role, client, service_id, python_function)
    # return python_function


def create_data_engineer(client=None):
    async def develop_python_functions(
        req, role: Role
    ) -> PythonOutput:
        """Complete python functions based on CodeEvalInput to pass the tests."""
        
        # if isinstance(req, SoftwareRequirement):
        func = await generate_code(req, role)
        # try:
        #     func = await test_run_python_function(role, client, req.id, func)
        # except RuntimeError as exp:
        #     req.additional_notes += f"\nPlease avoid the following error: {exp}"
        #     func = await generate_code(req, role)
        #     func = await test_run_python_function(role, client, req.id, func)

        return func


    data_engineer = Role(
        name="Alice",
        profile="Data Engineer",
        goal="Complete the python code script according to the code evaluation input, ensuring that it fulfills the desired functionality. Implement necessary algorithms, handle data processing, and write tests to validate the correctness of the function.",
        constraints=None,
        actions=[develop_python_functions],
    )
    return data_engineer

problems = read_problems()
DataEngineer = create_data_engineer()
ds = DataEngineer()
# get the first sample in problems
sub_problem = dict(random.sample(problems.items(), 1))
num_samples_per_task = 2

samples = [
    dict(task_id=task_id, completion=ds.develop_python_functions(sub_problem[task_id]["prompt"])['function_script'])
    for task_id in sub_problem
    for _ in range(num_samples_per_task)
]
write_jsonl(os.path.join("./.data","samples.jsonl"), samples)