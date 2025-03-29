import os
import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
from xml.etree import ElementTree as ET

class StudyConclusions(BaseModel):
    """The conclusions of the study"""
    content: str = Field(..., description="The conclusions of the study")

class StudyMethods(BaseModel):
    """The methods of the study"""
    content: str = Field(..., description="The methods of the study")

class StudySummary(BaseModel):
    """The summary of the study"""
    methods : StudyMethods = Field(..., description="The methods of the study")
    conclusions : StudyConclusions = Field(..., description="The conclusions of the study")

def create_summarizer():

    async def summarize_study(GEO_dir_path : str, role : Role = None) -> StudySummary:
        """Summarizes the study"""
        result = await role.aask(GEO_dir_path, StudySummary)
        return(result)

    
    summarizer = Role(
        name="Summarizer",
        profile="An agent that summarizes the study",
        goal="To summarize the study",
        constraints=None,
        model = "gpt-4-0125-preview",
        actions=[summarize_study],
    )
    return(summarizer)

def create_requirements_writer():

    def make_data_file_string(data_files : List[DataFile]) -> str:
        """Makes a string of the data files"""
        s = ""
        for df in data_files:
            s += f"Name : {df.name}\n"
            s += f"Description : {df.desc}\n"
            s += f"Location : {df.location}\n"
            s += f"Format : {df.format}\n\n"
        return s 
 
    async def write_requirements(analysis_options : AnalysisOptions, role : Role = None) -> None:
        """Writes the software requirements and specifications for all the analysis options"""
        tasks = [role.aask(hypothesis_workflow, WorkflowRequirements) for hypothesis_workflow in analysis_options.options]
        all_requirements = await asyncio.gather(*tasks)
        out_dir = "generated_files"
        os.makedirs(out_dir, exist_ok=True)
        for reqs in all_requirements:
            with open(os.path.join(out_dir, reqs.file_name), "w") as file:
                s = make_data_file_string(reqs.required_data_files)
                file.write(f"# Input Data Files\n{s}\n")
                s = '\n'.join([x for x in reqs.analysis_steps])
                file.write(f"# Analysis Steps\n{s}\n")
                file.write(f"# Desired Output\n{reqs.desired_output}\n")
        return
    
    requirements_writer = Role(
        name="Requirements Writer",
        profile="An agent that writes the software requirements for the hypotheses to investigate",
        goal="To write the requirements for the project",
        constraints=None,
        model = "gpt-4-0125-preview",
        actions=[write_requirements],
    )
    return(requirements_writer)

def make_team():
    agents = []
    data_scanner = create_data_scanner()
    agents.append(data_scanner)
    requirements_writer = create_requirements_writer()
    agents.append(requirements_writer)
    team = Team(name="NCBI GEO analyzers", profile="A team of agents meant to comprehensively understand and analyze a NCBI GEO repository", investment=0.7)
    team.hire(agents)
    return(team)

# Main function
async def main():
    team = make_team()
    user_request = "Test this"
    responses = await team.handle(Message(content=user_request, role="User"))
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()