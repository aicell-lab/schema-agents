import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Any

# Classes
class UserQuery(BaseModel):
    """The raw query input by the user."""
    user_query: str = Field(description="The raw query input by the user.")
    structured_query: str = Field(description="The validated and structured query ready for execution against the NCBI GEO database.")

class QueryResults(BaseModel):
    """Pretend you are a NCBI server, receive query from the user, and retrun a list of results retrieved from the NCBI GEO database."""
    query: UserQuery = Field(description="The structured query received from the InputHandler.")
    results: List[Any] = Field(description="The list of results retrieved from the NCBI GEO database.")

class Interpretation(BaseModel):
    """A textual summary of the findings from the data analysis."""
    results: QueryResults = Field(description="The results object containing data fetched by the GEOQueryAgent.")
    summary: str = Field(description="A textual summary of the findings from the data analysis.")
    patterns: List[Any] = Field(description="A list of identified patterns or significant data points within the results.")

class StudySuggestions(BaseModel):
    """A list of suggested follow-up studies based on the interpretation of the data."""
    interpretation: Interpretation = Field(description="The interpretation object containing the summary and patterns identified by the DataInterpreter.")
    suggestions: List[Any] = Field(description="A list of suggested follow-up studies based on the interpretation of the data.")

# Actions
async def receive_user_input(user_query : str, role: Role = None) -> UserQuery:
    """Receive input from the user, validate it, and convert it into a structured query."""
    # user_query = transform_input(user_query)
    structured_query = await role.aask(user_query, UserQuery)
    return structured_query

async def execute_geo_query(query : UserQuery, role: Role = None) -> QueryResults:
    """Execute the structured query against the NCBI GEO database and retrieve results."""
    results = await role.aask(query, QueryResults)
    
    return results

async def interpret_data(results : QueryResults, role: Role = None) -> Interpretation:
    """Analyze the query results and provide a summary with identified patterns."""
    # results = transform_input(results)
    interpretation = await role.aask(results, Interpretation)
    return interpretation

async def suggest_studies(interpretation : Interpretation, role: Role = None) -> StudySuggestions:
    """Generate suggestions for follow-up studies based on the interpreted data."""
    # interpretation = transform_input(interpretation)
    suggestions = await role.aask(interpretation, StudySuggestions)
    return suggestions

# Main function
async def main():
    input_handler = Role(
        name="InputHandler",
        profile="Responsible for receiving and validating user input, and converting it into a structured query for the NCBI GEO database.",
        goal="To ensure user input is correctly interpreted and formatted for querying the NCBI GEO database.",
        constraints=None,
        actions=[receive_user_input],
    )
    geo_query_agent = Role(
        name="GEOQueryAgent",
        profile="Executes structured queries against the NCBI GEO database and retrieves the results.",
        goal="To fetch relevant data from the NCBI GEO database based on the structured query provided by the InputHandler.",
        constraints=None,
        actions=[execute_geo_query],
    )
    data_interpreter = Role(
        name="DataInterpreter",
        profile="Analyzes the query results to extract meaningful insights and identify patterns.",
        goal="To interpret the data retrieved from the NCBI GEO database and provide a summary of the findings.",
        constraints=None,
        actions=[interpret_data],
    )
    study_suggester = Role(
        name="StudySuggester",
        profile="Generates recommendations for follow-up studies based on the interpreted data.",
        goal="To suggest potential follow-up studies that could be conducted based on the analysis of the NCBI GEO query results.",
        constraints=None,
        actions=[suggest_studies],
    )
    agents = [input_handler, geo_query_agent, data_interpreter, study_suggester]
    team = Team(name="GEO Query and Analysis Team", profile="A specialized team designed to interact with the NCBI GEO database, interpret the results, and suggest follow-up studies based on user input.", goal="To provide users with insightful analysis and actionable follow-up studies from NCBI GEO database queries.", investment=0.7)
    team.hire(agents)
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
    responses = await team.handle(Message(content="Mercury poisoning in humans", role="User"))
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
