import asyncio
from schema_agents.role import Role
from schema_agents.schema import Message
from pydantic import BaseModel, Field
from typing import List
from schema_agents.teams import Team



class SchemaActionOutline(BaseModel):
     """A detailed description of a python function containing type annotations, a function signature, and pseudocode for the function body"""
     content: str = Field(description = """The python function for performing the desired task. 
                          This should always contain a keyword argument called `role` of type `Role` that defaults to None. """)

class SchemaActionImplemented(BaseModel):
    """An implemented python function based off a function outline that uses built-in python libraries 
    to execute the desired action according to the pseudocode in the function outline body"""
    content: str = Field(description = """The finalized implemented python function. The function must follow the form
                         ```
                        async def function_name(input_arg : input_arg_type, role: Role = None) -> output_type:
                        inputs = <function_logic>
                        response = await role.aask(inputs, output_type)
                        return(response)
                         ```
                        the <function_logic> can interact with the input_arg by unpacking its fields and 
                         manipulating them or it can pass the input_arg directly into the `role.aask` function call
                         """)
     

class SchemaAgentOutline(BaseModel):
     """A autonomous agent capable of carrying out actions according to input and output schema"""
     name: str = Field(description = "The agent's name. It should refer to the agents desired function")
     profile: str = Field(description = "The agent's profile. It should give a brief description of the agent's desired function. This will be used as input to an LLM model")
     goal: str = Field(description = "The agent's goal. This will be used as input to an LLM model")
     actions: List[SchemaActionOutline] = Field(description = "The agent's list of actions")

class SchemaClass(BaseModel):
    """A schema definition for classes that will be passed between actions"""
    fields: List[str] = Field(description = "The class's fields. Each field must be type annotated and be of the form `field_name:field_type`")
    descriptions: List[str] = Field(description = "A detailed description of each of the class's fields")
 
class TeamOutline(BaseModel):
    """An outline for a team of agents to accomplish a given task"""
    name: str = Field(description="The name of the team")
    profile: str = Field(description="The profile of the team.")
    goal: str = Field(description="The goal for the team")
    agents: List[SchemaAgentOutline] = Field(description = "The agents involved in the task")
    classes: List[SchemaClass] = Field(description = """The classes passed between agents via actions. Each class consists of a list of fields that must be type annotated.
                                       """)
    actions: List[str] = Field(description = "The actions between each agent")


class ImplementedTeam(BaseModel):
    """A final version of the team"""
    name: str = Field(description="The name of the team")
    profile: str = Field(description="The profile of the team.")
    goal: str = Field(description="The goal for the team")
    agents: List[SchemaAgentOutline] = Field(description = "The agents involved in the task. Each agent")
    actions: List[SchemaActionImplemented] = Field(description = "The actions between each agent")
    classes: List[SchemaClass] = Field(description = "The classes passed between agents via actions")

class FinalCode(BaseModel):
    """A formatted string containing the python implementation of the team"""
    file_name: str = Field(description = "A filename to save the python file. Must have a .py extension")
    content: str = Field(description="""A formatted string to be pasted into a .py file which can run the team.
                         The string must start with the following import statements: 

                         ```import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from pydantic import BaseModel, Field```.
                        It must also import any relevant types from the python `typing` library that are used in the code.
                        After that there will be a section called "Classes" followed by the implemented team's classes where each class has the form:
                         ```
class <class_name>(BaseModel):
    <field>: <field_type> = Field(description = "<field_description>")
    ...
                         ``` where <field>, <field_type>, and <field_description> have been appropriately filled out.
                        After that there will be a section called "Actions" followed by the implemented team's actions.
                        After that there will be an asynchronous function called "main()". It will initialize the list of 
                        agents inside the main function in the following manner for each agent:
                         ```
<agent_name> = Role(
    name="<agent_name>",
    profile="<agent_profile>",
    goal="<agent_goal>",
    constraints=None,
    actions=[<agent_actions>],
)
                         ``` where <agent_name>, <agent_profile>, <agent_goal>, and <agent_actions> have been appropriately filled out. It will then add those agents to a list called `agents`. Then it will
                        create a variable `team` using the line 
                        `Team(name=<team_name>, profile=<team_profile>, goal=<team_goal>, investment = 0.7)` where the arguments
                         have been filled in appropriately. It will then contain the lines
                         ```
team.hire(agents)
event_bus = team.get_event_bus()
event_bus.register_default_events()
responses = await team.handle(Message(content="Mercury poisoning in humans", role="User"))
print(responses)```
                         And that will finish the main() function. Finally it will end with:
```if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()```
                         """)
    
async def write_final_code(implemented_team : ImplementedTeam, role: Role = None) -> FinalCode:
    """Take the implemented team and unpack it into a string which can be pasted into a .py file to run"""
    response = await role.aask(implemented_team, FinalCode)
    print(f"Saving file to filename: {response.file_name}")
    with open(response.file_name, 'w') as f:
        print(response.content, file = f)
    return(response)


async def make_team_outline(task: str, role: Role = None) -> TeamOutline:
     """Take the input task and design a team of autonomous agents that will carry out the task according to the user's needs"""
     response = await role.aask(task, TeamOutline)
     return(response)

async def implement_team(team_outline : TeamOutline, role: Role = None) -> ImplementedTeam:
     """Take the team outline and write an implemented team. The team outline consists of a name, profile, goal, """
    #  outline_agents = team_outline.agents
     response = await role.aask(team_outline, ImplementedTeam)
     return(response)


# async def write_team_code(team_plan : TeamOutline, role : Role = None) -> str:
#      """Take a team plan and write a python script that will run the team"""
#      script_code = f"""import asyncio
# from pydantic import BaseModel, Field
# from typing import List
# from schema_agents.role import Role
# from schema_agents.schema import Message
# from schema_agents.teams import Team
# from pydantic import BaseModel, Field\n"""

    
    
async def main():
    alice = Role(
        name="Alice",
        profile="Team Designer",
        goal="Your goal is to listen to user's request and propose an agent system for accomplishing this task.",
        constraints=None,
        actions=[make_team_outline],
    )

    bob = Role(
        name="bob",
        profile="Team Implementer",
        goal="Your goal is to take a team design and turn it into an implemented team",
        constraints=None,
        actions=[implement_team],
    )

    frank = Role(
        name = "frank",
        profile = "Final code writer",
        goal="Your goal is to take an implemented team and write a final output python file",
        constraints = None,
        actions = [write_final_code]
    )

    agents = [alice, bob, frank]
    team = Team(name="Assemblers", profile="Team assembler", goal="Take a user request and implement a finalized team", investment = 0.7)
    team.hire(agents)
    
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
    responses = await team.handle(Message(content="Design a team that will be able to take a user's input and query the NCBI GEO database using structured queries, interpret the results, and suggest follow up studies", role="User"))
    # with open("output.py", 'w') as f:
        # print(responses[2]["content"], file = f)
    print(responses)
    loop = asyncio.get_running_loop()
    loop.stop()
 
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()