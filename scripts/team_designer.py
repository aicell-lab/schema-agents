import asyncio
import os
from schema_agents.role import Role
from schema_agents.schema import Message
import pydantic
from pydantic import BaseModel, Field
from typing import List
from schema_agents.teams import Team
import json
from schema_agents.utils.pydantic import jsonschema_to_pydantic


class SchemaField(BaseModel):
    """A field within an ActionSchema"""
    field_name: str = Field(description = "The field's name")
    field_type: str = Field(description = "The field's type. Must be either a valid Python type or another ActionSchema defined in the team's list of action_schema.")

class ActionSchema(BaseModel):
    """A schema definition for classes that will be passed between actions"""
    name: str = Field(description = "The schema definition's name. It should follow python class naming conventions e.g. CamelCase")
    fields: List[SchemaField] = Field(description = "This ActionSchema's list of SchemaFields")
    field_descriptions: List[str] = Field(description = "A detailed description of each of the class's SchemaFields")
    class_description: str = Field(description = "A detailed description of this ActionSchema")

class Action(BaseModel):
    """An Action performed by Agents. Each Action takes either an ActionSchema or `str` as its input type."""
    name: str = Field(description = "The name of action. Should be formatted like a python function definition - lowercase with underscores in between words.")
    input_action_schema: str = Field(description = "The Action's input ActionSchema. Must either be a valid ActionSchema from the team's list of ActionSchema or `str`")
    output_action_schema: str = Field(description = "The Action's output ActionSchema. Must be a valid ActionSchema from the team's list of ActionSchema")
    code_logic: List[str] = Field(description = """The detailed logical steps involved in performing the desired action. Each step should be easily translatable to Python code.
    Each step may use the SchemaFields available in the Action's input_action_schema if helpful.""")

class Agent(BaseModel):
    """An autonomous agent capable of carrying out actions according to input and output ActionSchema."""
    name: str = Field(description = "The agent's name. It should refer to the agents desired function")
    profile: str = Field(description = "The agent's profile. It should give a brief description of the agent's desired function. This will be used as input to an LLM model")
    goal: str = Field(description = "The agent's goal. This will be used as input to an LLM model")
    actions: List[Action] = Field(description = "The agent's list of actions")

     
class AgentTeam(BaseModel):
    """A team of autonomous agents meant to accomplish a given task"""
    name: str = Field(description="The name of the team")
    profile: str = Field(description="The profile of the team.")
    goal: str = Field(description="The goal for the team")
    agents: List[Agent] = Field(description = "The agents involved in the task")
    action_schema: List[ActionSchema] = Field(description = "The list of all ActionSchema used in the team.")

class AgentTeamDraft(AgentTeam):
    """A team of autonomous agents such that (1) exactly one agent takes `str` as its input_action_schema for one of its Actions"""

    
class FinalResponse(BaseModel):
    """The end response of the team creation process"""
    content: str = Field(description="""The final response""")
    
async def write_final_code(implemented_team : AgentTeamDraft, role: Role = None) -> FinalResponse:
    """Take the implmeneted team, serialize it, and write it to JSON files"""
    team_dir = implemented_team.name.replace(' ', '_')
    class_dir = os.path.join(team_dir, 'action_schema')
    os.makedirs(class_dir, exist_ok = True)

    for a_s in implemented_team.action_schema:
        fields = {field.field_name : (field.field_type, Field(... , description=a_s.field_descriptions[i_field])) for i_field, field in enumerate(a_s.fields)}
        a_s_model = pydantic.create_model(a_s.name, **fields)
        a_s_model.__doc__ = a_s.class_description
        print(a_s.name)
        with open(os.path.join(class_dir, f"{a_s.name}.json"), 'w') as f:
            print(a_s_model.schema_json(), file = f)

    # with open(os.path.join(team_dir, "team.json"), "w") as f:
        # print(implemented_team.model_dump_json(), file = f)

    open(os.path.join(team_dir, '__init__.py'), 'w')
    # for action_schema in implemented_team.action_schema:

    # response = FinalResponse(content = str(implemented_team.model_dump_json()))
    response = FinalResponse(content = "I'm done")
    return(response)

async def draft_team(task: str, role: Role = None) -> AgentTeamDraft:
    """Take the input task and design a team of autonomous agents that will carry out the task according to the user's needs
    Each Action in the team should have a valid input_action_schema and output_action_schema. One and only one Action should have 
    `str` as its input_action_schema. This will be the first action performed by the team."""
    response = await role.aask(task, AgentTeamDraft)
    return(response)

    
async def main():
    agents = []

    alice = Role(
        name="Alice",
        profile="Team Designer",
        goal="Your goal is to listen to user's request and propose an agent system for accomplishing this task.",
        constraints=None,
        actions = [draft_team]
    )
    agents.append(alice)

    # bob = Role(
    #     name="bob",
    #     profile="Team Implementer",
    #     goal="Your goal is to take a team design and turn it into an implemented team",
    #     constraints=None,
    #     actions=[implement_team],
    # )
    # agents.append(bob)

    frank = Role(
        name = "frank",
        profile = "Final code writer",
        goal="Your goal is to take an implemented team and write a final output json file",
        constraints = None,
        actions = [write_final_code]
    )
    agents.append(frank)


    team = Team(name="Assemblers", profile="Team assembler", goal="Take a user request and implement a finalized team", investment = 0.7)
    team.hire(agents)
    
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
    # responses = await team.handle(Message(content="Design a team that will be able to take a user's input and query the NCBI GEO database using structured queries, interpret the results, and suggest follow up studies", role="User"))
    responses = await team.handle(Message(content="Design a two-agent team that will come up with recipes given a list of ingredients.", role="User"))
    # print(responses)
    # loop = asyncio.get_running_loop()
    # loop.stop()
 
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()