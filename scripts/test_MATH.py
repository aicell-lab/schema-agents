import random
import os
import json
import asyncio
import glob
from itertools import zip_longest
from typing import List, Union, Optional, Dict, Any
from pydantic import BaseModel, Field

from schema_agents.tools.code_interpreter import create_mock_client
from schema_agents.schema import Message
from schema_agents.role import Role

class MathInput(BaseModel):
    """Represents a math problem."""
    problem: str = Field(..., description="Math problem.")
    type: str = Field(..., description="Type of the math problem.")
    level: str = Field(..., description="Level of the math problem.")
    
class MathOutput(BaseModel):
    """Represents the solution of a math problem."""
    solution: str = Field(..., description="Step by step solution of the math problem.")
    final_answer: str = Field(..., description="Final answer of the math problem.")


dataroot = "../MATH/test/*/*.json"
all_filenames = glob.glob(dataroot)
problem_set = []
for fname in all_filenames:
    with open(fname, 'r') as fp:
        try:
            problem_data = json.load(fp)
        except Exception as e:
            print(f"Error loading JSON from {fname}", e)
            raise e
    problem_set.append(problem_data)

print(f"Loaded {len(problem_set)} samples.")

async def solve_math_problem(problem: MathInput, role: Role) -> MathOutput:
    """Solve the math problem based on the given input."""
    inputs = [problem.problem] + [problem.type] + [problem.level]
    response = await role.aask(inputs, MathOutput)
    return response

math_expert = Role(
        name="Alice",
        profile="Math Expert",
        goal="Carefully read the math problem and solve it step by step. Simplify the solution as much as possible, write down the solution and the final answer.",
        constraints=None,
        actions=[solve_math_problem],
    )

async def main():
    event_bus = math_expert.get_event_bus()
    event_bus.register_default_events()
    problem = random.choice(problem_set)
    p = MathInput(problem=problem['problem'], type=problem['type'], level=problem['level'])
    response = await math_expert.handle(Message(content=p.json(), data=p, role="User"))
    script = response[-1].data.function_script
    return script

if __name__ == "__main__":
    asyncio.run(main())