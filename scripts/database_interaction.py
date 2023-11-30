import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict

# Classes
class StructuredQuery(BaseModel):
    """A query formatted to search the NCBI GEO Database inspired by the user's input query"""
    query: str = Field(description = "The NCBI GEO query string, it MUST fit the NCBI GEO database search syntax. The query should focus only on the Geo DataSets database.")
    query_url: str = Field(description = "The query convert into a NCBI E-utils url. It must be only the url and it must folow the E-utils syntax")
    # search_terms: List[str] = Field(description = "A list of key terms extracted from the user input that are relevant to the search.")
    # query_parameters: Dict[str, str] = Field(description = "A dictionary of parameters that will be used to structure the query for the database.")

class QueryResults(BaseModel):
    """The XML results from a NCBI GEO database query using the Entrez E-utils API"""
    results: str = Field(description = "The XML string content of the returned results")

async def interpret_input(user_input: str, role: Role = None) -> StructuredQuery:
    """Takes the user inputm, parses it, and creates an NCBI GEO structured query utilizing the databae's syntax"""
    # structured_query = {'search_terms': [], 'query_parameters': {}}
    # structured_query['search_terms'], structured_query['query_parameters'] = await role.aask(user_input, (list, dict))
    # return StructuredQuery(**structured_query)
    result = await role.aask(user_input, StructuredQuery)
    return(result)    


# class QueryResults(BaseModel):
#     search_results: List[Dict] = Field(description = "A list of results retrieved from the database query.")

# class InterpretedResults(BaseModel):
#     significant_data_points: List[Dict] = Field(description = "A list of data points identified as significant from the query results.")
#     summary: str = Field(description = "A textual summary of the findings from the results interpretation.")

# class StudySuggestions(BaseModel):
#     suggested_studies: List[str] = Field(description = "A list of potential follow-up studies suggested based on the interpreted results.")

# Actions
async def interpret_input(user_input: str, role: Role = None) -> StructuredQuery:
    """Takes the user inputm, parses it, and creates an NCBI GEO structured query utilizing the databae's syntax"""
    # structured_query = {'search_terms': [], 'query_parameters': {}}
    # structured_query['search_terms'], structured_query['query_parameters'] = await role.aask(user_input, (list, dict))
    # return StructuredQuery(**structured_query)
    result = await role.aask(user_input, StructuredQuery)
    return(result)

# async def execute_query(query: StructuredQuery, role: Role = None) -> QueryResults:
#     """Takes a structured query, converts it into a url following the """

# async def interpret_results(results: QueryResults, role: Role = None) -> InterpretedResults:
#     interpreted_results = {'significant_data_points': [], 'summary': ''}
#     interpreted_results['significant_data_points'], interpreted_results['summary'] = await role.aask(results, (list, str))
#     return InterpretedResults(**interpreted_results)

# async def suggest_studies(interpreted_results: InterpretedResults, role: Role = None) -> StudySuggestions:
#     suggested_studies = await role.aask(interpreted_results, list)
#     return StudySuggestions(suggested_studies=suggested_studies)

# Main function
async def main():
    agents = []
    input_interpreter = Role(
        name="InputInterpreter",
        profile="An agent that processes user input to form structured queries for the NCBI GEO database.",
        goal="To accurately interpret user input and convert it into a structured query for the NCBI GEO database.",
        constraints=None,
        actions=[interpret_input],
    )
    agents.append(input_interpreter)
    
    # database_querier = Role(
    #     name="DatabaseQuerier",
    #     profile="An agent responsible for executing structured queries against the NCBI GEO database and retrieving results.",
    #     goal="To perform database queries efficiently and retrieve accurate and relevant data.",
    #     constraints=None,
    #     actions=[execute_query],
    # )
    # results_interpreter = Role(
    #     name="ResultsInterpreter",
    #     profile="An agent that analyzes the query results from the NCBI GEO database to extract meaningful insights.",
    #     goal="To interpret the search results and provide a clear summary of the findings.",
    #     constraints=None,
    #     actions=[interpret_results],
    # )
    # study_suggester = Role(
    #     name="StudySuggester",
    #     profile="An agent that uses the interpreted results to suggest potential follow-up studies.",
    #     goal="To provide users with actionable suggestions for further research based on the analyzed data.",
    #     constraints=None,
    #     actions=[suggest_studies],
    # )

    # agents = [input_interpreter, database_querier, results_interpreter, study_suggester]
    team = Team(name="GEO Research Assistant Team", profile="A team specialized in querying the NCBI GEO database, interpreting the results, and suggesting follow-up studies based on user input.", goal="To assist users in conducting research by providing an efficient way to access and analyze data from the NCBI GEO database and to suggest relevant follow-up studies.", investment=0.7)

    team.hire(agents)
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
    user_request = "I'm interested in studying how lead poisoning works"
    responses = await team.handle(Message(content=user_request, role="User"))
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()