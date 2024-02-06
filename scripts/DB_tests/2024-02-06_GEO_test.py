import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
from xml.etree import ElementTree as ET

class DataFile(BaseModel):
    """A file in the data repository coming from a sample"""
    name: str = Field(..., description="The name of the file")
    desc: str = Field(..., description="A description of the file")
    # biosample_url: str = Field(..., description="The biosample url of the sample the file came from")
    # geo_sample_id : str = Field(..., description="The GEO sample id of the sample the file came from")

class Sample(BaseModel):
    """A sample in the data repository"""
    # geo_sample_id: str = Field(..., description="The GEO sample id of the sample")
    # name: str = Field(..., description="The name of the sample")
    desc: str = Field(..., description="A description of the sample")
    data_files: List[DataFile] = Field(..., description="A list of the data files associated with the sample")

class DataDescription(BaseModel):
    """A description of all the data available in the repository as demarcated by a <Supplementary-Data> tag inside of a <Sample> tag in the matrix file"""
    # data_files: list[DataFile] = Field(..., description="A detailed description of the data available in the repository")
    samples: list[Sample] = Field(..., description="A detailed description of the samples available in the repository")

class HypothesisWorkflow(BaseModel):
    """The workflow for testing a hypothesis"""
    hypothesis : str = Field(..., description="The hypothesis to test")
    workflow: list[str] = Field(..., description="The workflow for testing a hypothesis")

class AnalysisOptions(BaseModel):
    """The possible hypotheses you could test using ONLY the data in the repository"""
    options: list[HypothesisWorkflow] = Field(..., description="All the possible hypotheses you could test and how to test them in pseudo-code format")

class PseudoWorkflow(HypothesisWorkflow):
    """A workflow in pseudo-code format"""
    workflow: str = Field(..., description="A workflow in precise pseudo-code format. EVERY step must be included, including installing dependencies and downloading data")

class PseudoWorkflows(BaseModel):
    """The analysis options in pseudo-code format"""
    options: list[PseudoWorkflow] = Field(..., description="A list of pseudocode-workflows")
    

async def scan_matrix(matrix_path : str, role : Role = None) -> DataDescription:
    """Scans the matrix for available data"""
    with open(matrix_path, "r") as file:
        matrix_content = file.read()
    result = await role.aask(matrix_content, DataDescription)
    return(result)

async def evaluate_analysis_options(data_description : DataDescription, role : Role = None) -> AnalysisOptions:
    """Evaluates the analysis options"""
    result = await role.aask(data_description, AnalysisOptions)
    for hypothesis in result.options:
        print(hypothesis)
    return(result)

async def write_pseudo_workflows(analysis_options : AnalysisOptions, role : Role = None) -> PseudoWorkflows:
    """Writes the analysis options in pseudo-code format"""
    # pseudo_workflows = []
    tasks = [role.aask(hypothesis_workflow, PseudoWorkflow) for hypothesis_workflow in analysis_options.options]
    pseudo_workflows = await asyncio.gather(*tasks)
    # for hypothesis in analysis_options.options:
        # pseudo_workflow = await role.aask(hypothesis, PseudoWorkflow)
        # pseudo_workflows.append(pseudo_workflow.workflow)
    return(PseudoWorkflows(options=pseudo_workflows))

class WorkflowScript(BaseModel):
    """A workflow script"""
    filename: str = Field(..., description="The filename of the script. It should contain no spaces (only underscores) and end in .nf")
    script: str = Field(..., description="The script for the workflow in nextflow format")

class WorkflowScripts(BaseModel):
    """A list of workflow scripts"""
    scripts: List[WorkflowScript] = Field(..., description="A list of workflow scripts")

async def write_workflow_scripts(pseudo_workflows : PseudoWorkflows, role : Role = None) -> WorkflowScripts:
    """Writes the workflow scripts to test the proposed hypotheses"""
    # nextflow_scripts = []
    # for hypothesis in analysis_options.options:
    #     nextflow_script = await role.aask(hypothesis, WorkflowScript)
    #     nextflow_scripts.append(nextflow_script)
    #     with open(nextflow_script.filename, "w") as file:
    #         file.write(nextflow_script.script)
    tasks = [role.aask(pseudocode_workflow, WorkflowScript) for pseudocode_workflow in pseudo_workflows.options]
    nextflow_scripts = await asyncio.gather(*tasks)
    for nextflow_script in nextflow_scripts:
        with open(nextflow_script.filename, "w") as file:
            file.write(nextflow_script.script)
    return(WorkflowScripts(scripts=nextflow_scripts))

class ScriptReview(WorkflowScript):
    """A workflow script that has been reviewed for completeness and correctness"""
    comments : str = Field(..., description="The additions that must be made to the script to make it complete and correct")
    
class ScriptReviews(BaseModel):
    """A list of script reviews"""
    reviews: List[ScriptReview] = Field(..., description="A list of script reviews")

async def review_scripts(workflow_scripts : WorkflowScripts, role : Role = None) -> ScriptReviews:
    """Read over each script and check if it is completely self-contained and executable from start to finish. If not, add the necessary code to make it so. If it is, simply copy the script"""
    # script_reviews = []
    # for script in workflow_scripts.scripts:
    #     with open(script.filename, 'r') as file:
    #         script_contents = file.read()
    #         script_review = await role.aask(script_contents, ScriptReview)
    #     script_reviews.append(script_review)
    tasks = [role.aask(script, ScriptReview) for script in workflow_scripts.scripts]
    script_reviews = await asyncio.gather(*tasks)
    for script_review in script_reviews:
        with open(script_review.filename, "w") as file:
            file.write(script_review.script)
    return(ScriptReviews(reviews=script_reviews))

class RevisedScript(WorkflowScript):
    """A workflow script revised to be COMPLETELY self-contained and executable, including installing dependencies and filling out any missing code"""
    pass

async def revise_scripts(script_reviews : ScriptReviews, role : Role = None) -> None:
    """Revise each script to be completely self-contained and executable using the script reviews"""
    # for script in script_reviews.reviews:
    #     revised_script = await role.aask(script, RevisedScript)
    #     with open(revised_script.filename, "w") as file:
    #         file.write(revised_script.script)
    tasks = [role.aask(script, RevisedScript) for script in script_reviews.reviews]
    revised_scripts = await asyncio.gather(*tasks)
    for revised_script in revised_scripts:
        with open(revised_script.filename, "w") as file:
            file.write(revised_script.script)


def make_team():
    agents = []
    data_scanner = Role(
        name="Data Scanner",
        profile="An agent that scans for available data",
        goal="To comprehensively scan for available data",
        constraints=None,
        model = "gpt-4-0125-preview",
        actions=[scan_matrix, evaluate_analysis_options],
    )
    agents.append(data_scanner)

    code_writer = Role(
        name="Code Writer",
        profile="An agent that writes code",
        goal="To write code that is complete, self-contained, and error-free",
        constraints=None,
        model = "gpt-4-0125-preview",
        actions=[write_pseudo_workflows, write_workflow_scripts, revise_scripts],
    )
    agents.append(code_writer)

    code_reviewer = Role(
        name="Code Reviewer",
        profile="An agent that reviews code",
        goal="To check that the code is completely self-contained and executable and to suggest necessary changes to make it so",
        constraints=None,
        model = "gpt-4-0125-preview",
        actions=[review_scripts],
    )
    agents.append(code_reviewer)

    team = Team(name="NCBI GEO analyzers", profile="A team of agents meant to comprehensively understand and analyze a NCBI GEO repository", investment=0.7)

    team.hire(agents)
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
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