import time
import sys
import os
import asyncio
import extensions
from pydantic import BaseModel, Field
from typing import Optional
from schema_agents.role import Role
from schema_agents.schema import Message

class ResponseStep(BaseModel):
    """Response step"""
    name : str = Field(description = "Step Name")
    details : Optional[dict] = Field(None, description = "Step Details")

class RichResponse(BaseModel):
    """Rich response with text and intermediate steps"""
    text : str = Field(..., description = "Response text")
    steps : Optional[list[ResponseStep]] = Field(None, description = "Intermediate steps")

def create_assistants():

    builtin_extensions = extensions.get_builtin_extensions()
    
    async def respond(
            query : str, role : Role = None
    ) -> RichResponse:
        """Answers the user's question directory or retrieve relevant information, or create a Python Script to get information about details of models."""
        steps = []
        inputs = [query]
        # inputs = query
        tools = []
        for extension in builtin_extensions:
            tool = await extensions.extension_to_tool(extension)
            tools.append(tool)
        
        class ThoughtsSchema(BaseModel):
            """Details about the thoughts"""
            reasoning: str = Field(..., description="reasoning and constructive self-criticism; make it short and concise in less than 20 words")
        
        response, metadata = await role.acall(inputs,
                                              tools,
                                              return_metadata=True,
                                              thoughts_schema=ThoughtsSchema,
                                              max_loop_count = 10)
        # I need this line to avoid a script crash. Without some print reference to `metadata` or `response`, script complains of an unexpected string output
        with open(os.devnull, 'w') as devnull:
            print(response, file=devnull)
        result_steps = metadata['steps']
        for idx, step_list in enumerate(result_steps):
            steps.append(
                ResponseStep(
                    name = f"step-{idx}",
                    details = {"details" : extensions.convert_to_dict(step_list)}
                )
            )
        return RichResponse(text = response, steps = steps)
    
    assistant = Role(
        instructions = "You are the assistant, a helpful agent for helping the user",
        actions = [respond],
        model = "gpt-4-0125-preview",
    )
    event_bus = assistant.get_event_bus()
    event_bus.register_default_events()
    all_extensions = [{"name" : ext.name, "description" : ext.description} for ext in builtin_extensions]
    return [{"name" : "assistant", "agent" : assistant, "extensions" : all_extensions}]



async def main():
    assistant = create_assistants()[0]['agent']
    # event_bus = alice.get_event_bus()
    # event_bus.register_default_events()
    user_query = "Make a plan to find NCBI datasets relevant to digital microfluidics using the NCBI eutils API, then execute it to find the datasets."
    responses = await assistant.handle(Message(content=user_query, role="User"))
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()


