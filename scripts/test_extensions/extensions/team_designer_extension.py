import asyncio
from paperqa import Docs
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from bioimageio_chatbot.utils import ChatbotExtension
import asyncio
from schema_agents.role import Role, Message


class SchemaActionOutline(BaseModel):
    """An action performed by an agent. An action takes either a string or a `SchemaClass` as input and returns a string or a `SchemaClass` as output"""
    name : str = Field(description = "The name of the action. It contains NO spaces and NO special characters. It should be written in snake_case format.")
    input_type: Union[str, BaseModel] = Field(description = """The input type for the action. This can be a string or a `SchemaClass`""")
    output_type: Union[str, BaseModel] = Field(description = """The output type for the action. This can be a string or a `SchemaClass`""")
    description: str = Field(description = "A description of the action")



class SchemaAgentOutline(BaseModel):
     """A autonomous agent capable of carrying out actions according to input and output schema"""
     name: str = Field(description = "The agent's name. It should refer to the agents desired function")
     profile: str = Field(description = "The agent's profile. It should give a brief description of the agent's desired function. This will be used as input to an LLM model")
     goal: str = Field(description = "The agent's goal. This will be used as input to an LLM model")
     actions: List[SchemaActionOutline] = Field(description = "The agent's list of actions")

class SchemaClass(BaseModel):
    """A schema definition for classes that will be passed between actions"""
    fields: List[str] = Field(description = "The class's fields. Each field must be type annotated and be of the form `field_name:field_type`")
    field_descriptions: List[str] = Field(description = "A detailed description of each of the class's fields")
    class_description: str = Field(description = "A detailed description of the class")

# class TeamOutline(BaseModel):
#     """An outline for a team of agents to accomplish a given task"""
#     name: str = Field(description="The name of the team")
#     profile: str = Field(description="The profile of the team.")
#     goal: str = Field(description="The goal for the team")
#     agents: List[SchemaAgentOutline] = Field(description = "The agents involved in the task")
#     classes: List[SchemaClass] = Field(description = """The classes passed between agents via actions. Each class consists of a list of fields that must be type annotated.""")
#     actions: List[SchemaActionOutline] = Field(description = "The actions between each agent")
    
class TeamOutline(BaseModel):
    """An outline for a team of agents to accomplish a given task"""
    name: str = Field(description="The name of the team")
    profile: str = Field(description="The profile of the team.")
    goal: str = Field(description="The goal for the team")
    actions: List[SchemaActionOutline] = Field(description = "The sequential list of actions between the agents that will solve the task")


async def make_team_outline(task: str, role: Role = None) -> TeamOutline:
     """Take the input task and design a team of autonomous agents that will carry out the task according to the user's needs"""
     response = await role.aask(task, TeamOutline)
     return(response)


async def run_extension(query : str) -> TeamOutline:
    """Take the input task and design a team of autonomous agents that will carry out the task according to the user's needs"""
    assistant = Role(
        instructions = "You are a team creator, your role is to create a team of agents to accomplish a given task",
        actions = [make_team_outline],
        model = "gpt-4-0125-preview",
    )
    team_outline = await make_team_outline(query, assistant)
    return team_outline

def get_extensions():
    return [
        ChatbotExtension(
            name="team_designer",
            description="Create a team of agents equipped with tools to execute complex tasks",
            execute=run_extension,
        )
    ]


async def main():
    # assistant = Role(
    #     instructions = "You are the assistant, a helpful agent for helping the user",
    #     actions = [make_team_outline],
    #     model = "gpt-4-0125-preview",
    # )
    # event_bus = assistant.get_event_bus()
    # event_bus.register_default_events()
    # user_query = """Tell me how many calories are in an apple"""
    # responses = await assistant.handle(Message(content=user_query, role="User"))

    team_outline = await run_extension("Design a team that will solve the following problem: `There are 3 animals in a room, 2 leave. How many are left?`")
    print(team_outline)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()


