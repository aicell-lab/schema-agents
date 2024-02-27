from schema_agents.provider.openai_api import retry
import asyncio
from paperqa import Docs
from typing import List, Optional, Union, Type, Any, get_type_hints, Tuple
from pydantic import BaseModel, Field, validator, create_model
from bioimageio_chatbot.utils import ChatbotExtension
import asyncio
from schema_agents.role import Role, Message
import json




class ThoughtsSchema(BaseModel):
    """Details about the thoughts"""
    reasoning: str = Field(..., description="reasoning and constructive self-criticism; make it short and concise in less than 20 words")

class SchemaAgentOutline(BaseModel):
    """A autonomous agent capable of carrying out actions according to input and output schema"""
    name: str = Field(description = "The agent's name. It should refer to the agents desired function")
    instructions : str = Field(description = "The agent's instructions. It should give a brief description of the agent's role, background, desired functions, and actions. This will be used as input to an LLM model")
    
class TeamOutline(BaseModel):
    """An outline for a team of agents to accomplish a given task"""
    name: str = Field(description="The name of the team")
    profile: str = Field(description="The profile of the team.")
    goal: str = Field(description="The goal for the team")
    agents: list[SchemaAgentOutline] = Field(description = "The agents involved in the task")


class SchemaField(BaseModel):
    """A PyDantic model for a field that will be passed into an action or returned from an action"""
    field_name: str = Field(description = "The name of the field. It should refer to the field of the object e.g. `name`, `age`, `weight`, `height`")
    field_type: str = Field(description = "The type of the field. It should refer to the type of the object e.g. `str`, `int`, `float`, `list`, `dict`, `List[int]`, `Dict[str, int]`, etc. It MUST be a type that can be used in PyDantic models.")
    field_description : str = Field(description = "A detailed description of the field and its type")

class SchemaClassDraft(BaseModel):
    """Represents a PyDantic model (extending PyDantic's BaseModel) for a class that will either be passed into an action or returned from an action"""
    class_name: str = Field(description = "The name of the class. It should be written in CamelCase format e.g. `ObjectName`")
    class_description: str = Field(description = "A detailed description of the class and its Fields. Each field has a name and a type. ")
    # fields : list[str] = Field(description = "The fields of the class")

class SchemaClassImplemented(SchemaClassDraft):
    """Represents a PyDantic model (extending PyDantic's BaseModel) for a class that will either be passed into an action or returned from an action"""
    fields : list[SchemaField] = Field(description = "The fields of the class. You MUST populate this list, it cannot be empty. Each field has a name, a type, and a description.")

class SchemaActionOutline(BaseModel):
    """An action performed by an agent. An action takes either a string or a `SchemaClass` as input and returns a string or a `SchemaClass` as output"""
    name : str = Field(description = "The name of the action function")
    agent : SchemaAgentOutline = Field(description = "The agent who performs the action")
    description : str = Field(description = "A detailed description of the action including what it does, how it does it, and what it returns. It should also include the input and output schema for the action.")
    input_schema : SchemaClassDraft = Field(description = "The input schema for the action")
    output_schema : SchemaClassDraft = Field(description = "The output schema for the action")

class SchemaActionImplemented(SchemaActionOutline):
    """An action performed by an agent. An action takes either a string or a `SchemaClass` as input and returns a string or a `SchemaClass` as output"""
    input_schema : SchemaClassImplemented = Field(description = "The input schema for the action")
    output_schema : SchemaClassImplemented = Field(description = "The output schema for the action")
    
class FlowOutline(BaseModel):
    """An outline for a flow of actions to accomplish a given task"""
    steps : list[SchemaActionOutline] = Field(description = "The sequential list of actions between the agents that will solve the task. The output_schema of one action MUST match the input_schema of the next action.")

class FlowImplemented(BaseModel):
    """A flow of actions to accomplish a given task"""
    steps : list[SchemaActionImplemented] = Field(description = "The sequential list of actions between the agents that will solve the task")

def get_python_type(type_str):
    return {
        "str" : str,
        "int" : int,
        "float" : float,
        "bool" : bool,
        "datetime" : datetime,
        "List[str]" : List[str],
        "List[int]" : List[int],
        "List[float]" : List[float],
        "List[bool]" : List[bool],
        "List[datetime]" : List[datetime],
        "Optional[str]" : Optional[str],
        "Optional[int]" : Optional[int],
        "Optional[float]" : Optional[float],
        "Optional[bool]" : Optional[bool],
        "Optional[datetime]" : Optional[datetime],
    }.get(type_str, str)

def create_pydantic_model_from_schema(schema: SchemaClassImplemented) -> Type[BaseModel]:
    data = json.loads(schema.json())
    fields = {field['field_name']: (get_python_type(field['field_type']), Field(description=field['field_description'])) for field in data['fields']}
    DynamicModel = create_model(data['class_name'], __doc__ = data['class_description'], **fields)
    return DynamicModel

async def make_team_outline(task: str, role: Role = None) -> TeamOutline:
     """Take the input task and design a team of autonomous agents that will carry out the task according to the user's needs"""
     response = await role.aask(task, TeamOutline)
     return(response)

async def make_flow(team_outline: TeamOutline, role: Role = None) -> FlowOutline:
    """Take the input team outline and design a flow of actions that will solve the task using the agents in the team outline"""
    response = await role.aask(team_outline, FlowOutline)
    return(response)

@retry(5)
async def run_extension(query : str) -> TeamOutline:
    """Take the input task and design a team of autonomous agents that will carry out the task according to the user's needs"""
    assistant = Role(
        instructions = "You are a team creator, your role is to create a team of agents to accomplish a given task",
        actions = [make_team_outline],
        model = "gpt-4-0125-preview",
    )
    event_bus = assistant.get_event_bus()
    event_bus.register_default_events()
    team_outline = await assistant.aask(query, TeamOutline)
    with open('team.json', 'w') as f:
        json.dump(json.loads(team_outline.json()), f, ensure_ascii = False, indent=4)
    print(team_outline)
    flow_outline = await assistant.aask(team_outline, FlowOutline)
    with open('flow.json', 'w') as f:
        json.dump(json.loads(flow_outline.json()), f, ensure_ascii = False, indent=4)

    implemented_steps = []
    schema_drafts = {}
    for step in flow_outline.steps:
        for sd in [step.input_schema, step.output_schema]:
            if sd.class_name not in schema_drafts:
                schema_drafts[sd.class_name] = sd
    
    # Use gather syntax to make the requests concurrently
    implemented_schemas = await asyncio.gather(*[assistant.aask(sd, SchemaClassImplemented) for sd in schema_drafts.values()])
    implemented_schemas = {sd.class_name : sd for sd in implemented_schemas}

    # implemented_schemas = {}
    # for sd in schema_drafts.values():
        # implemented_schema = await assistant.aask(sd, SchemaClassImplemented)
        # implemented_schemas[sd.class_name] = implemented_schema
    for step in flow_outline.steps:
        implemented_step = SchemaActionImplemented(
            name = step.name,
            agent = step.agent,
            description = step.description,
            input_schema = implemented_schemas[step.input_schema.class_name],
            output_schema = implemented_schemas[step.output_schema.class_name]
        )
        implemented_steps.append(implemented_step)
    flow_implemented = FlowImplemented(steps = implemented_steps)
    with open('flow_implemented.json', 'w') as f:
        json.dump(json.loads(flow_implemented.json()), f, ensure_ascii = False, indent=4)
    
    team_instance = {}
    for agent in team_outline.agents:
        agent_instance = Role(
            name = agent.name,
            instructions = f"{agent.instructions}",
            model = "gpt-4-0125-preview",
        )
        team_instance[agent.name] = agent_instance
    # async def execute_flow(implemented_fow : FlowImplemented):
    for step in flow_implemented.steps:
        agent_name = step.agent.name
        call_fun = team_instance[agent_name].acall(step.description, tools = [], output_schema = step.output_schema, thoughts_schema = ThoughtsSchema, max_loop_count = 10, return_metadata = True)
        # FIXME: need to change step.output_schema to an actual instantiated schema type
        response, metadata = await call_fun


            # response = await agent.aask(step.input_schema, step.output_schema)
            # with open(f'{step.name}.json', 'w') as f:
                # json.dump(json.loads(response.json()), f, ensure_ascii = False, indent=4)
            # print(response)
    # team_outline = await assistant.acall(query, tools = [], output_schema=TeamOutline, thoughts_schema=ThoughtsSchema, max_loop_count = 10)
    # flow_outline = await assistant.acall(team_outline, tools = [], output_schema=FlowOutline, thoughts_schema=ThoughtsSchema, max_loop_count = 10)
    return flow_outline



def get_extensions():
    return [
        ChatbotExtension(
            name="team_designer",
            description="Create a team of agents equipped with tools to execute complex tasks",
            execute=run_extension,
        )
    ]




async def main():
    # team_outline = await run_extension("Design a team that will solve the following problem: `There are 3 animals in a room, 2 leave. How many are left?`")
    team_outline = await run_extension("Design a team that will solve the following problem: `I have a PDF file of a scientific paper. I want to understand and reproduce the results from the paper.`")
    #######################################################
    # Call acall on the agent instances
    #######################################################
    

    print('\n\nDone\n\n')

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()






