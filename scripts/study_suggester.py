import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
import requests
from xml.etree import ElementTree as ET
import re
from collections import defaultdict
from typing import Union
import os

class StructuredQuery(BaseModel):
    """A query formatted to search the NCBI PubMed Database inspired by the user's input query"""
    query: str = Field(description = "The NCBI PubMed query string, it MUST fit the NCBI PubMed database search syntax.")
    query_url: str = Field(description = "The query converted into a NCBI E-utils url. It must be only the url and it must folow the E-utils syntax. It should specify xml return mode.")

class PaperFetcherUrl(BaseModel):
    """A NCBI PubMed E-utils efetch url to get the abstracts PubMed. The url will be constructed by looking at the IDs present in the xml search results given here"""
    query_url: str = Field(description="The NCBI E-utils url for using efetch to get the PubMed abstracts from the IDs")

class RelevantTexts(BaseModel):
    """The found relevant texts including their titles, authors, affiliations, and abstracts"""
    titles: list[str] = Field(description="The paper titles")
    authors: list[Union[list[str], str]] = Field(description="The paper authors")
    affiliations: list[Union[list[str], str]] = Field(description="The affilications of the paper authors")
    abstracts: list[str] = Field(description="The abstract text for each paper")

async def interpret_input(user_input: str, role: Role = None) -> StructuredQuery:
    """Takes the user input, parses it, and creates an NCBI GEO structured query utilizing the databae's syntax"""
    result = await role.aask(user_input, StructuredQuery)
    return(result)   

async def submit_request(url: str, out_file: str | None = None) -> str:
    response = requests.get(url)
    assert response.status_code == 200
    content = response.content.decode('utf-8')
    if out_file:
        with open(out_file, 'w') as f:
            print(content, file = f)
    return(content)

async def extract_article_info(xml_content):
    # Parsing the XML content
    root = ET.fromstring(xml_content)
    # This will hold all the extracted article information
    articles_info = []
    # Iterate over each article in the XML
    for article in root.findall('.//PubmedArticle'):
        # Initialize dictionary to store information about the article
        article_info = {'title': None, 'authors': [], 'affiliations': [], 'abstract': None}
        # Extracting the title
        title_element = article.find('.//ArticleTitle')
        if title_element is not None:
            article_info['title'] = title_element.text
        # Extracting authors and affiliations
        for author in article.findall('.//Author'):
            author_name = ' '.join(filter(None, [author.findtext('LastName'), author.findtext('ForeName'), author.findtext('Initials')]))
            if author_name:
                article_info['authors'].append(author_name)
            affiliation = author.findtext('.//AffiliationInfo/Affiliation')
            if affiliation and affiliation not in article_info['affiliations']:
                article_info['affiliations'].append(affiliation)
        # Extracting the abstract
        abstract_text = []
        for abstract in article.findall('.//Abstract/AbstractText'):
            if abstract.text:
                abstract_text.append(abstract.text)
        article_info['abstract'] = ' '.join(abstract_text)
        # Adding the extracted information to the list
        articles_info.append(article_info)
    return articles_info


async def get_relevant_texts(structured_query: StructuredQuery, role: Role = None) -> PaperFetcherUrl:
    """Takes the structured query and creates an E-utils url for searching the NCBI PubMed database using the IDs"""
    id_search_results = await submit_request(structured_query.query_url)
    paper_fetcher_url = await role.aask(id_search_results, PaperFetcherUrl)
    return(paper_fetcher_url)

async def scrape_paper_info(paper_fetcher_url: PaperFetcherUrl, role: Role = None) -> RelevantTexts:
    """Takes the paper fetcher url, searches the PubMed database for the papers and scrapes relevant paper information"""
    search_results_dir = "./study_suggester_intermediate_files"
    os.makedirs(search_results_dir, exist_ok=True)
    search_results_file = os.path.join(search_results_dir, "pubmed_search_results.xml")
    abstract_search_results = await submit_request(paper_fetcher_url.query_url, out_file = search_results_file)
    paper_info = await extract_article_info(abstract_search_results)
    ap = {}
    for d in paper_info:
        for k,v in d.items():
            if k not in ap:
                ap[k] = []
            ap[k].append(v)
    relevant_texts = RelevantTexts(
        titles = ap['title'], 
        authors = ap['authors'], 
        affiliations=ap['affiliations'], 
        abstracts=ap['abstract'])
    papers_file = os.path.join(search_results_dir, "relevant_papers.txt")
    with open(papers_file, 'w') as f:
        print(f"{relevant_texts.titles}\n\n", file = f)
        print(f"{relevant_texts.authors}\n\n", file = f)
        print(f"{relevant_texts.affiliations}\n\n", file = f)
        print(f"{relevant_texts.abstracts}\n\n", file = f)
    return(relevant_texts)

class SuggestedStudies(BaseModel):
    """Three suggested follow up experiments"""
    experiment_names: list[str] = Field(description="A list of names for each experiment")
    experiment_material: list[list[str]] = Field(description="A list of materials required for each experiment")
    experiment_expected_results: list[str] = Field(description = "A list of expected outcomes for each experiment")
    experiment_protocols: list[list[str]] = Field(description = "The protocol steps for each experiment")
    # experiment_diagrams: list[str] = Field(description = "Graphviz code for a diagram illustrating the workflow for each experiment")

class SuggestedStudiesWithDiagrams(SuggestedStudies):
    """Three suggested follow up experiments with a simple diagram written in mermaid.js for each one showing what expected data will look like"""
    experiment_data_diagrams: list[str] = Field(description = "The code for a mermaid.js diagram (either a XYChart, Pie, or QuadrantChart) showing what the expected data results would look like for the experiment")

class SummaryWebsite(BaseModel):
    """A summary single-page webpage written in html that neatly presents the suggested studies for user review"""
    html_code: str = Field(description = "The html code for a single page website summarizing the information in the suggested studies appropriately including the diagrams")

async def illustrate_studies(suggested_studies: SuggestedStudies, role: Role = None) -> SuggestedStudiesWithDiagrams:
    """Takes suggested studies and creates a mermaid.js diagram for each one displaying what the output data might look like"""
    result = await role.aask(suggested_studies, SuggestedStudiesWithDiagrams)
    return(result)

async def suggest_studies(relevant_texts: RelevantTexts, role: Role = None) -> SuggestedStudies:
    """Takes found papers, reads their abstracts and suggests possible follow up studies"""
    result = await role.aask(relevant_texts.abstracts, SuggestedStudies)
    return(result)

async def write_website(suggested_studies: SuggestedStudiesWithDiagrams, role: Role = None) -> SuggestedStudies:
    """Takes the suggested studies and creates an html page neatly summarizing the information in the suggested studies"""
    result = await role.aask(suggested_studies, SummaryWebsite)
    with open('summary.html', 'w') as f:
        print(result.html_code, file = f)
    return(result)



# Main function
async def main():
    agents = []
    input_interpreter = Role(
        name="InputInterpreter",
        profile="An agent that processes user input to scrape user interests as well as equipment.",
        goal="To accurately interpret user input and find terms relevant to the user's interests and equipment",
        constraints=None,
        actions=[interpret_input],
    )
    agents.append(input_interpreter)

    database_querier = Role(
        name="DatabaseQuerier",
        profile="An agent that queries the PubMed database, find the relevant IDs, then queries the database again using the IDs to get relevant paper information",
        goal="To find the relevant papers on PubMed and extract the paper information including titles, authors, affiliations, and abstracts.",
        actions=[get_relevant_texts, scrape_paper_info]
    )
    agents.append(database_querier)

    study_suggester = Role(
        name = "StudySuggester",
        profile="An expert scientist agent that reads found abstracts and suggests possible follow up studies",
        goal = "To suggest follow up studies that will yield interesting results relevant to the found papers",
        actions = [suggest_studies]
    )
    agents.append(study_suggester)

    study_illustrator = Role(
        name = "StudyIllustrator",
        profile="An expert data scientist and illustrator that reads suggested studies, creating diagrams of what study data results might look like",
        goal = "To create example diagrams of what experiment outputs might look like",
        actions = [illustrate_studies]
    )
    agents.append(study_illustrator)

    website_writer = Role(
        name = "WebsiteWriter",
        profile = "An expert html and UI developer who creates a single-page html webpage to present and summarize the suggested studies",
        goal = "To create a beautiful single-page html website that will summarize the suggested studies",
        actions = [write_website]
    )
    agents.append(website_writer)
    

    # agents = [input_interpreter, database_querier, results_interpreter, study_suggester]
    team = Team(name="Study Proposers", profile="A team specialized in parsing user inputs, searching for relevant papers on PubMed, and suggesting follow up studies.", investment=0.7)

    team.hire(agents)
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
    user_request = "I'm interested in studying the effect of lead poisoning on mammalian cells. I have a BSL2 lab with a fume hood, liquid handling robot, incubators, shakers, and standard "
    responses = await team.handle(Message(content=user_request, role="User"))
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()