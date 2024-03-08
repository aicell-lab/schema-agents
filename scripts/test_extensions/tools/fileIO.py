from pydantic import Field
from schema_agents import schema_tool, Role

@schema_tool
async def write_to_file(string : str = Field(description="The string to write to a file"),
                        out_file : str = Field(description="The file to write to")) -> str:
    """Write a string to a file. Use this when output are too large to write to stdout or as a return string to the user"""
    with open(out_file, 'w') as f:
        f.write(string)
    return "Success"

@schema_tool
async def read_file(string : str = Field(description="The file to read from")) -> str:
    """Read a file and return the string"""
    with open(string, 'r') as f:
        return f.read()
    
