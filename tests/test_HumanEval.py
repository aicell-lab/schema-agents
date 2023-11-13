
import random
import os
import json
from typing import List, Union, Optional, Dict, Any

from pydantic import BaseModel, Field
from human_eval.evaluation import evaluate_functional_correctness
from human_eval.data import write_jsonl, read_problems

from schema_agents.tools.code_interpreter import create_mock_client
from schema_agents.schema import Message
from schema_agents.teams.image_analysis_hub.schemas import PythonFunctionScript
from schema_agents.role import Role

class PythonOutput(BaseModel):
    """Represents a Python function with all its properties."""
    function_names: List[str] = Field(..., description="Function names in the script.")
    function_script: str = Field(..., description="Completed python function script. This script should be able to pass the test cases. Includes imports, function definition, logic, and implementation.")
    docstring: Optional[str] = Field(None, description="Brief notes for usage, debugging, potential error fixing, and further improvements.")
   



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



async def generate_code(
    req, role: Role
) -> PythonOutput:
    """Complete python functions based on the given input."""
    # retrieve req-related memories from long term memory
    response = await role.aask(req, PythonOutput)
    return response

async def develop_python_functions(
    req: str, role: Role
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



async def main():
    # Your existing code...

    problems = read_problems()
    test_path = "/home/alalulu/workspace/schema-agents/tests/data"
    
    selected_problems = dict(random.sample(problems.items(), 20))
    write_jsonl(os.path.join(test_path, "selected_problems.jsonl.gz"), selected_problems.values())
    
    sub_problem = read_problems(os.path.join(test_path, "selected_problems.jsonl.gz"))

    num_samples_per_task = 1
    samples = await asyncio.gather(*[
        generate_sample(task_id, sub_problem[task_id]["prompt"], num_samples_per_task)
        for task_id in sub_problem
    ])

    write_jsonl(os.path.join(test_path, "samples.jsonl"), samples)
    await evaluate_functional_correctness("/home/alalulu/workspace/schema-agents/tests/data/samples.jsonl", [1], problem_file=os.path.join(test_path, "selected_problems.jsonl.gz"))

async def generate_sample(task_id, prompt, num_samples):
    data_engineer = Role(
        name="Alice",
        profile="Data Engineer",
        goal="Complete the python code script according to the code evaluation input, ensuring that it fulfills the desired functionality. Implement necessary algorithms, handle data processing, and write tests to validate the correctness of the function.",
        constraints=None,
        actions=[develop_python_functions],
    )

    async def generate_one_completion(prompt):
        response = await data_engineer.handle(Message(content=prompt, role="User"))
        script = response[-1].data.function_script
        return script

    return {
        "task_id": task_id,
        "completion": await generate_one_completion(prompt)
    }
if __name__ == "__main__":
    # test_path = "/home/alalulu/workspace/schema-agents/tests/data"
    
    # import asyncio
    # asyncio.run(main())
    problems = read_problems()
    def check_correctness():
        # Importing multiprocessing only when this function is called
        import multiprocessing
        manager = multiprocessing.Manager()
        test_path = "/home/alalulu/workspace/schema-agents/tests/data"
        # human_path = "/home/alalulu/workspace/human-eval/data"
        print(evaluate_functional_correctness(os.path.join(test_path,"samples.jsonl"), [1], problem_file=os.path.join(test_path, "selected_problems.jsonl.gz")))
    check_correctness()