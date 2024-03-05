from pydantic import BaseModel, Field, ConfigDict
from langchain_community.vectorstores.faiss import FAISS

class File(BaseModel):
    """A file"""
    name: str = Field(description="The name of the file")
    path: str = Field(description="The path to the file")

class FileFormatDescription(BaseModel):
    """A file format description"""
    description: str = Field(description="A detailed description of the file's contents and its format")

class DataFile(File):
    """A data file"""
    path: str = Field(description="The path to the file")
    data_description: str = Field(description="A description of the data in the file")
    # format: FileFormatDescription = Field(description="A detailed description of the format of the file")

class PublicDataSet(BaseModel):
    """A public data set"""
    name: str = Field(..., description="The name of the data set")
    summary : str = Field(..., description="A summary of the data set")
    data_generation_summary : str = Field(..., description="A summary of how the data was generated")
    samples_description: str = Field(..., description="A detailed description of the samples in the data set")
    data_files : list[DataFile] = Field(..., description="A list of the data files in the data set and a description of each file")

class ScrapedGEOPage(PublicDataSet):
    """A page scraped from the NCBI GEO database"""
    pass

class GEODataSet(PublicDataSet):
    """A data set from the NCBI GEO database"""
    geo_accession: str = Field(description="The GEO accession number of the data set")

class Hypothesis(BaseModel):
    """A testable hypothesis"""
    file_name : str = Field(..., description="The name of the workflow file, it should contain no spaces, briefly refer to the hypothesis, and end in .md")
    hypothesis: str = Field(..., description="The hypothesis")
    test_workflow : str = Field(..., description="The workflow for testing the hypothesis")

class Hypotheses(BaseModel):
    """A list of hypotheses to test"""
    hypotheses: list[Hypothesis] = Field(..., description="A list of hypotheses")

class InformaticHypothesis(Hypothesis, PublicDataSet):
    """A data set and a hypothesis to test using that data set"""
    pass

class InformaticHypotheses(Hypotheses):
    """A list of hypotheses to test"""
    hypotheses: list[InformaticHypothesis] = Field(..., description="A list of hypotheses")




    

class MainFindings(BaseModel):
    """The main findings of a paper"""
    main_findings: str = Field(..., description="The main findings of the paper")
                               
class ComputationalMethods(BaseModel):
    """The computational methods used in a paper"""
    computational_methods: str = Field(..., description="The computational methods used in the paper")

class ExperimentalMethods(BaseModel):
    """The experimental methods used in a paper"""
    experimental_methods: str = Field(..., description="The experimental methods used in the paper")

class Samples(BaseModel):
    """The samples used in a paper"""
    samples: str = Field(..., description="The samples used in the paper")

class PaperSummary(BaseModel):
    """A summary of a paper"""
    # topics: str = Field(..., description="The keyword topics of the paper")
    main_findings: MainFindings = Field(..., description="The main findings of the paper")
    computational_methods: ComputationalMethods = Field(..., description="The computational methods used in the paper")
    experimental_methods: ExperimentalMethods = Field(..., description="The experimental methods used in the paper")
    samples: Samples = Field(..., description="The samples used in the paper")