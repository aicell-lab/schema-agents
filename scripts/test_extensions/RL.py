import json
from typing import List
import time
import sys
import os
import asyncio
import extensions
from pydantic import BaseModel, Field
from typing import Optional
from schema_agents.role import Role
from schema_agents.schema import Message
from extensions.paperqa_extension import Paper
import extensions.team_designer_extension as team_designer_extension

class ThoughtsSchema(BaseModel):
    """Details about the thoughts"""
    reasoning: str = Field(..., description="reasoning and constructive self-criticism; make it short and concise in less than 20 words")

class ResponseStep(BaseModel):
    """Response step"""
    name : str = Field(description = "Step Name")
    details : Optional[dict] = Field(None, description = "Step Details")

class RichResponse(BaseModel):
    """Rich response with text and intermediate steps"""
    text : str = Field(..., description = "Response text")
    steps : Optional[list[ResponseStep]] = Field(None, description = "Intermediate steps")

def create_assistants():
    async def respond(query : str, role : Role = None) -> RichResponse:
        """Answers the user's question directory or create a team to solve the task, monitor its results, and revise the team if necessary."""
        draft_team = await team_designer_extension.run_extension(query)
        

        steps = []
        inputs = [query]
        tools = [team_designer_extension.run_extension]
        response, metadata = await role.acall(inputs,
                                              tools,
                                              return_metadata=True,
                                              thoughts_schema=ThoughtsSchema,
                                              max_loop_count = 10)
        result_steps = metadata['steps']
        for idx, step_list in enumerate(result_steps):
            steps.append(
                ResponseStep(
                    name = f"step-{idx}",
                    details = {"details" : extensions.convert_to_dict(step_list)}
                )
            )
        return RichResponse(text = response, steps = steps)
    
    manager = Role(
        instructions = "You are the project manager, you will be provided a user request and you will be responsible for creating a team of agents to accomplish the task. You will then try to complete the task using the team you created. If the task is not completed properly by the team, you will revise the team and try again using what you have learned.",
        actions = [respond],
        model = "gpt-4-0125-preview",
    )
    event_bus = manager.get_event_bus()
    event_bus.register_default_events()
    return manager

async def main():
    # manager = create_assistants()[0]['agent']
    # user_query = """Make a detailed plan for reproducing the paper located at '/Users/gkreder/gdrive/exponential-chain/GSE254364/405.pdf'. 
    # Use as many calls to websearch and paperqa as necessary to get all the information you need. The plan should be EXTREMELY detailed, it should include all the detailed steps for how to access data, download packages, and run code"""
    # user_query = """Tell me how many calories are in an apple"""
    # responses = await manager.handle(Message(content=user_query, role="User"))

    # manager = create_assistants()[0]['agent']
    manager = create_assistants()
    # user_query = "Make a team that will read a paper and make a detailed plan for reproducing the paper."
    user_query = """Make a team and use it solve the following problem: `Understand and reproduce the results from the paper located at '/Users/gkreder/gdrive/exponential-chain/GSE254364/405.pdf'.`"""
    responses = await manager.handle(Message(content=user_query, role="User"))

    # manager = create_assistants()[0]['agent']
    # user_query = "Make a detailed plan for reproducing the paper. Use as many calls to websearch and paperqa as necessary to get all the information you need"
    # user_data = Paper(location = '/Users/gkreder/Downloads/2024-02-01_exponential_chain/GSE254364/405.pdf', location_type = "file")
    # responses = await manager.handle(Message(content=user_query, data=user_data, role="User"))
    print(responses)
    print('\n\n\n')
    with open('responses.json', 'w') as f:
        json.dump(json.loads(responses[0].content), f, ensure_ascii = False, indent=4)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()


