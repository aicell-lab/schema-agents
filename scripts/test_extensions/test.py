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

class ThoughtsSchema(BaseModel):
    """Details about the thoughts"""
    reasoning: str = Field(..., description="reasoning and constructive self-criticism; make it short and concise in less than 20 words")

class PaperResults(BaseModel):
    """A summary of a scientific paper"""
    key_topics: List[str] = Field(description="The keyword topics of the paper")
    is_informatic : bool = Field(description="Whether the paper involved informatic, computational, or data analysis that involves coding")
    key_findings : List[str] = Field(description="The key findings of the paper")
    methods : List[str] = Field(description="The methods used in the paper")

class ReproductionPlan(BaseModel):
    """A plan to reproduce the results of a paper"""
    plan : str = Field(description="The plan to reproduce the results of the paper")

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

    async def get_all_paper_info(paper : Paper, role: Role = None) -> RichResponse:
        """Make then execute a plan to get all the information about a scientific paper necessary to reproduce its results. Use a combination of websearch and paperqa as many times as necessary to get all the information you need"""
        steps = []
        inputs = [paper]
        tools = []
        for extension in builtin_extensions:
            tool = await extensions.extension_to_tool(extension)
            tools.append(tool)
        
        response, metadata = await role.acall(inputs,
                                              tools,
                                              return_metadata=True,
                                              thoughts_schema=ThoughtsSchema,
                                              max_loop_count = 10)
        with open(os.devnull, 'w') as devnull:
            print(response, file=devnull)
        # return response
        result_steps = metadata['steps']
        for idx, step_list in enumerate(result_steps):
            steps.append(
                ResponseStep(
                    name = f"step-{idx}",
                    details = {"details" : extensions.convert_to_dict(step_list)}
                )
            )
        return RichResponse(text = response, steps = steps)
        
        
    
    async def respond(query : str, role : Role = None) -> RichResponse:
        """Answers the user's question directory or retrieve relevant information, or create a Python Script to get information about details of models."""
        steps = []
        inputs = [query]
        # inputs = query
        tools = []
        for extension in builtin_extensions:
            tool = await extensions.extension_to_tool(extension)
            tools.append(tool)
        
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
    
    paper_summarizer = Role(
        instructions = "You are the paper summarizer, a helpful agent for summarizing scientific papers and finding all the information necessary to reproduce the results",
        actions = [get_all_paper_info],
        model = "gpt-4-0125-preview",
    )
    
    assistant = Role(
        instructions = "You are the assistant, a helpful agent for helping the user",
        actions = [respond],
        model = "gpt-4-0125-preview",
    )
    event_bus = assistant.get_event_bus()
    event_bus.register_default_events()
    all_extensions = [{"name" : ext.name, "description" : ext.description} for ext in builtin_extensions]
    return [{"name" : "assistant", "agent" : assistant, "extensions" : all_extensions}]
    # return [{"name" : "assistant", "agent" : paper_summarizer, "extensions" : all_extensions}]



async def main():
    assistant = create_assistants()[0]['agent']
    # user_query = """Make a detailed plan for reproducing the paper located at '/Users/gkreder/gdrive/exponential-chain/GSE254364/405.pdf'. 
    # Use as many calls to websearch and paperqa as necessary to get all the information you need. The plan should be EXTREMELY detailed, it should include all the detailed steps for how to access data, download packages, and run code"""
    user_query = """Make a detailed plan for reproducing the paper located at '/Users/gkreder/gdrive/exponential-chain/GSE254364/405.pdf'. Your final output should be in the form of a nextflow script. 
    Use as many calls to websearch and paperqa as necessary to get all the information you need. The plan should be EXTREMELY detailed, it should include all the detailed steps for how to access data, download packages, and run code"""
    responses = await assistant.handle(Message(content=user_query, role="User"))

    # assistant = create_assistants()[0]['agent']
    # user_query = "Make a detailed plan for reproducing the paper. Use as many calls to websearch and paperqa as necessary to get all the information you need"
    # user_data = Paper(location = '/Users/gkreder/Downloads/2024-02-01_exponential_chain/GSE254364/405.pdf', location_type = "file")
    # responses = await assistant.handle(Message(content=user_query, data=user_data, role="User"))
    print(responses)
    print('\n\n\n')
    with open('responses.json', 'w') as f:
        json.dump(json.loads(responses[0].content), f, ensure_ascii = False, indent=4)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()


