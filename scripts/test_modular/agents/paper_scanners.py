from pydantic import BaseModel, Field
from schema_agents.role import Role
from schemas import *

class PaperScanner(Role):
    """An agent for scanning and processing papers"""
    def __init__(self):
        super().__init__(
            name="Paper Scanner",
            profile="An agent for scanning and processing scientific papers",
            goal="To scan and process scientific papers",
            constraints=None,
            actions=[self.scan],
        )

    async def scan(self, matrix_file_path: str, role: Role = None) -> PublicDataSet:
        """Scans a database"""
        with open(matrix_file_path, "r") as file:
            matrix_file_contents = file.read()
        result = await role.aask(matrix_file_contents, GEODataSet)
        return(result)