
import random
import os
import json
import asyncio
from itertools import zip_longest
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

async def main():
    
    problems = read_problems()
    test_path = "./data/HumanEval"
    selected_problems = {}
    # selected_problems = dict(random.sample(problems.items(), 100))
    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)
    
    for key, value in problems.items():
        # Extract the numeric part of the key
        key_number = int(key.split('/')[1])
        # Check if the numeric part is between 0 and 100 (inclusive)
        if 160 <= key_number <= 164:
            selected_problems[key] = value
    # write_jsonl(os.path.join(test_path, "selected_problems.jsonl.gz"), selected_problems.values())
    # sub_problem = read_problems(os.path.join(test_path, "selected_problems.jsonl.gz"))
    problem_set_chunks = grouper(selected_problems.items(), 5)
    
    num_samples_per_task = 10
    for chunk in problem_set_chunks:
        # remove None values in the chunk
        chunk = [x for x in chunk if x is not None]
        chunk = {k: v for k, v in chunk}
        samples = await asyncio.gather(*[
            generate_sample(task_id, chunk[task_id]["prompt"], num_samples_per_task)
            for task_id in chunk
        ])
        write_jsonl(os.path.join(test_path, f"iteration{num_samples_per_task}_samples.jsonl"), samples, append=True)
    # await evaluate_functional_correctness("/home/alalulu/workspace/schema-agents/tests/data/samples.jsonl", [1], problem_file=os.path.join(test_path, "selected_problems.jsonl.gz"))


if __name__ == "__main__":
    # asyncio.run(main())
    test_path = "./data/HumanEval"
    problems = read_problems()
    write_jsonl(os.path.join(test_path, "problems.jsonl.gz"), problems.values())
    def check_correctness():
        # Importing multiprocessing only when this function is called
        import multiprocessing
        manager = multiprocessing.Manager()
        
        # human_path = "/home/alalulu/workspace/human-eval/data"
        print(evaluate_functional_correctness(os.path.join(test_path,"iteration10_samples.jsonl"), [1], problem_file=os.path.join(test_path, "problems.jsonl.gz")))
    check_correctness()