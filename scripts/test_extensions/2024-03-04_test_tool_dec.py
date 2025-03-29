from enum import Enum
import inspect
from schema_agents.provider.openai_api import retry
import asyncio
from paperqa import Docs
from typing import List, Optional, Union, Type, Any, get_type_hints, Tuple, Literal
from pydantic import BaseModel, Field, validator, create_model
from bioimageio_chatbot.utils import ChatbotExtension
import asyncio
from schema_agents.role import Message
from schema_agents import schema_tool, Role
import json
# from .extensions.paperqa_tool import ask_pdf_paper

class LocationType(str, Enum):
    file = "file"
    url = "url"
        
class Paper(BaseModel):
    """A scientific journal or conference article to study"""
    location : str = Field(description="The location of the paper, either a file path or a url")
    location_type : LocationType = Field(description="The type of file location it MUST be either 'file' or 'url'")
    question :  Optional[str] = Field(None, description="The question to ask about the paper")

@schema_tool
async def ask_pdf_paper(location : str = Field(description="The location of the paper, either a file path or a url"),
                        location_type : LocationType = Field(description="The type of file location it MUST be either 'file' or 'url'"),
                        question :  str = Field(description="The question to ask about the paper")) -> str:
    """Query a paper for information"""
    docs = Docs()
    if location_type == LocationType.file:
        await docs.aadd(location)
    elif location_type == LocationType.url:
        await docs.add_url(location)
    else:
        raise ValueError(f"Invalid location type: {location_type}")
    answer = await aux_query(question, docs)
    with open(os.devnull, 'w') as devnull:
        print(answer, file=devnull)
    return answer

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
    original_query : str = Field(description="The original query that the team is designed to solve")
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


def expose_tool(function):
    name = function.__name__
    signature = inspect.signature(function)
    docstring = function.__doc__
    info = f"Tool Name: `{name}`\n\nTool Usage: {docstring}"
    return info

# TOOLS = [ask_pdf_paper]
# TOOL_DICT = {tool.__name__ : tool for tool in TOOLS}
# TOOL_DESC = '\n----------\n'.join([expose_tool(x) for x in TOOLS])

# # Create an enum for listing out tools
# class ToolTup(Enum):
#     """The name of the tool"""
#     ask_pdf_paper = ("ask_pdf_paper", """Query a paper for information""")



# class AskPDF(BaseModel):
    # """A tool used by an agent to accomplish tasks involving extracting information from PDFs of scientific papers"""
    # name : str = Field(description = "The name of the tool")
    # description : str = Field(description = "A detailed description of the tool and its usage")


# class AgentToolChoice(BaseModel):
#     """A tool used by an agent to accomplish a given task"""
#     choice : list[AgentTool] = Field(description = "The choice of tool used by the agent to accomplish the task. If no tools besides a base LLM are needed, this is simply an empty list `[]`")

class SchemaActionOutline(BaseModel):
    """An action performed by an agent. An action takes either a string or a `SchemaClass` as input and returns a string or a `SchemaClass` as output"""
    name : str = Field(description = "The name of the action function")
    agent : SchemaAgentOutline = Field(description = "The agent who performs the action")
    description : str = Field(description = "A detailed description of the action including what it does, how it does it, and what it returns. It should also include the input and output schema for the action.")
    input_schema : SchemaClassDraft = Field(description = "The input schema for the action")
    output_schema : SchemaClassDraft = Field(description = "The output schema for the action")

# class AgentTool(Enum):
    # """A tool used by an agent to accomplish a given task. Each tool is presented as a string of the form `TOOL_NAME : TOOL_DESCRIPTION`. For example, `ask_pdf_paper : Query a paper for information` is a tool called `ask_pdf_paper` that queries a paper for information."""
    # ask_pdf_paper = 'ask_pdf_paper : Query a paper for information'

class AgentTool(BaseModel):
    """A tool used by an agent to accomplish a given task"""
    pass
class AskPDF(AgentTool):
    """A tool used by an agent to accomplish tasks involving extracting information from PDFs of scientific papers"""
    pass

class TestTool(AgentTool):
    """A useless tool that isn't used by any agent to accomplish any task"""
    pass 

TOOL_DICT = {AskPDF : ask_pdf_paper,
             TestTool : print}

class ToolChoices(BaseModel):
    """A choice of tools used by an agent to accomplish a given action"""
    choices : list[Union[AskPDF, TestTool]] = Field(description = "The choice of tools used by the agent to accomplish the task. ONLY choose tools that might be helpful for the given task beyond the agent's base LLM capabilities. If no extra tools are needed, this is simply an empty list `[]`.")

class SchemaActionImplemented(SchemaActionOutline):
    """An action performed by an agent. An action takes either a string or a `SchemaClass` as input and returns a string or a `SchemaClass` as output"""
    input_schema : SchemaClassImplemented = Field(description = "The input schema for the action")
    output_schema : SchemaClassImplemented = Field(description = "The output schema for the action")
    tools : Optional[list[AgentTool]] = Field(None, description = "The choice of tools used by the agent to accomplish the task. If no tools besides a base LLM are needed, this is simply an empty list `[]`")
    
class FlowOutline(BaseModel):
    """An outline for a flow of actions to accomplish a given task"""
    steps : list[SchemaActionOutline] = Field(description = "The sequential list of actions between the agents that will solve the task. The output_schema of one action MUST match the input_schema of the next action.")

class FlowImplemented(BaseModel):
    """A flow of actions to accomplish a given task"""
    steps : list[SchemaActionImplemented] = Field(description = "The sequential list of actions between the agents that will solve the task")

def get_python_type(type_str):
    from datetime import datetime 
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
     response.original_query = task
     return(response)

@retry(3)
async def make_flow(team_outline: TeamOutline, role: Role = None) -> FlowOutline:
    """Take the input team outline and design a flow of actions that will solve the task using the agents in the team outline"""
    response = await role.aask(team_outline, FlowOutline)
    return(response)

# @retry(5)
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
    # flow_outline = await assistant.aask(team_outline, FlowOutline)
    flow_outline = await assistant._run_action(make_flow, Message(content = "Design a flow", data = team_outline))
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

    for step in flow_outline.steps:
        implemented_step = SchemaActionImplemented(
            name = step.name,
            agent = step.agent,
            description = step.description,
            input_schema = implemented_schemas[step.input_schema.class_name],
            output_schema = implemented_schemas[step.output_schema.class_name],
        )
        implemented_steps.append(implemented_step)
    tool_choices = await asyncio.gather(*[assistant.aask(sai, ToolChoices) for sai in implemented_steps])
    tool_choices_flattened = [list({type(tool) : tool for tool in tools.choices}.values()) for tools in tool_choices]
    tool_choices_fun = [[TOOL_DICT[type(tool)] for tool in tools] for tools in tool_choices_flattened]
    for i_sai, sai in enumerate(implemented_steps):
        sai.tools = tool_choices_fun[i_sai]
    flow_implemented = FlowImplemented(steps = implemented_steps)
    with open('flow_implemented.json', 'w') as f:
        json.dump(json.loads(flow_implemented.json()), f, ensure_ascii = False, indent=4)
    return flow_implemented, team_outline

def get_extensions():
    return [
        ChatbotExtension(
            name="team_designer",
            description="Create a team of agents equipped with tools to execute complex tasks",
            execute=run_extension,
        )
    ]
async def get_pydantic_model(schema: SchemaClassImplemented) -> Type[BaseModel]:
    try:
        schema_type = create_pydantic_model_from_schema(schema)
    except Exception as e:
        print(e)
        pydantic_assistant = Role(
            name = "PydanticAssistant",
            instructions = "You are the pydantic model creator, your role is to create a pydantic model from a schema",
            model = "gpt-4-0125-preview")
        res = await pydantic_assistant.aask(f"Failed to instantiate schema from SchemaClassImplemented: {schema}. Please try generating again. Error was the following: {str(e)}", SchemaClassImplemented)
        schema_type = create_pydantic_model_from_schema(res)
    
    return schema_type

async def create_initial_input(user_input : str, execution_flow : FlowImplemented) -> Type[BaseModel]:

    schema_type = create_pydantic_model_from_schema(execution_flow.steps[0].input_schema)
    input_preparer = Role(
        name = "InputPreparer",
        instructions = "You are the input preparer, your role is to prepare the input for the first action in the execution flow",
        model = "gpt-4-0125-preview",
    )
    response = await input_preparer.aask(user_input, schema_type)


async def run_flow(execution_flow : FlowImplemented, team_outline : TeamOutline) -> str:
    team_instance = {}
    for agent in team_outline.agents:
        agent_instance = Role(
            name = agent.name,
            instructions = f"{agent.instructions}",
            model = "gpt-4-0125-preview",
        )
        team_instance[agent.name] = agent_instance
    # async def execute_flow(implemented_fow : FlowImplemented):
    flow_object = team_outline.original_query
    flow_objects = []
    for step in execution_flow.steps:
        agent_name = step.agent.name
        schema_type = await get_pydantic_model(step.output_schema)
        call_fun = team_instance[agent_name].acall([flow_object, step.description], tools = step.tools, output_schema = schema_type, thoughts_schema = ThoughtsSchema, max_loop_count = 10, return_metadata = True)
        response, metadata = await call_fun
        flow_object = response
        flow_objects.append(response)
    return response, flow_objects


async def main():
    # team_outline = await run_extension("Design a team that will solve the following problem: `There are 3 animals in a room, 2 leave. How many are left?`")
    execution_flow, team_outline = await run_extension("Design a team that will solve the following problem: `I have a PDF file of a scientific paper located at /Users/gkreder/gdrive/exponential-chain/GSE254364/405.pdf. I want to understand and reproduce the results from the paper.`")
    run_result = await run_flow(execution_flow, team_outline)
    print(run_result)
    #######################################################
    # Call acall on the agent instances
    #######################################################
    

    print('\n\nDone\n\n')

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()






