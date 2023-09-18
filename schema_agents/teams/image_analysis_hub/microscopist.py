import asyncio
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.tools.code_interpreter import create_mock_client


class MovePosition(BaseModel):
    """Position for the microscope to move to."""
    x: float = Field(default=0.0, description="x position")
    y: float = Field(default=0.0, description="y position")
    z: float = Field(default=0.0, description="z position")

class SnapConfig(BaseModel):
    """Configuration for snapping an image."""
    path: str = Field(default="", description="save image path")
    exposure: float = Field(default=0.0, description="exposure time")
    
class MultiDimensionalAcquisitionConfig(BaseModel):
    """Configuration for multi-dimensional acquisition."""
    path: str = Field(default="", description="save images path")
    timeout: float = Field(default=0.0, description="timeout")
    query: str = Field(default="", description="user's original request for acquiring multi-dimensional images")
    plan: str = Field(default="", description="plan for acquiring multi-dimensional images")

class MultiDimensionalAcquisitionScript(BaseModel):
    """Python script for multi-dimensional acquisition.
    In the script, you can use the following functions to control the microscope:
    - `microscope_move({'x': 0.0, 'y': 0.0, 'z': 0.0})`
    - `microscope_snap({'path': './images', 'exposure': 0.0})`
    """
    script: str = Field(default="", description="Script for acquiring multi-dimensional images")
    explanation: str = Field(default="", description="Brief explanation for the script")

class ExecutionResult(BaseModel):
    """Result of executing a Python script."""
    status: str = Field(description="Status of executing the script")
    outputs: List[Dict[str, Any]] = Field(default=[], description="Outputs of executing the script")
    traceback: Optional[str] = Field(default=None, description="Traceback of executing the script")

INIT_SCRIPT = """
def microscope_move(position):
    print(f"===> Moving to: {position}")

def microscope_snap(config):
    print(f"===> Snapped an image with exposure {config['exposure']} and saved to: { config['path']}")
"""

class Microscope():
    def __init__(self, client):
        self.client = client
        self.initialized = False
    
    async def move(self, position: MovePosition=None):
        """Move the objective to a position."""
        if not self.initialized:
            await self.client.execute_code(INIT_SCRIPT)
            self.initialized = True
        print("Moving to: "+str(position))
        self.client.execute_code(f"microscope_move({position.dict()})")
        
    async def snap(self, config: SnapConfig=None):
        """Snap an image from the microscope."""
        if not self.initialized:
            await self.client.execute_code(INIT_SCRIPT)
            self.initialized = True
        print("save image to: " + config.path)
    
    async def plan(self, query: str=None, role: Role=None) -> Union[SnapConfig, MovePosition, MultiDimensionalAcquisitionConfig]:
        """Make a plan for image acquisition tasks."""
        return await role.aask(query, MultiDimensionalAcquisitionConfig)
        
    async def multi_dimensional_acquisition(self, config: MultiDimensionalAcquisitionConfig=None, role: Role=None) -> ExecutionResult:
        """Execute complex multi-dimensional image acquisition requests by using Python script."""
        if not self.initialized:
            await self.client.execute_code(INIT_SCRIPT)
            self.initialized = True
        print("Acquiring images in multiple dimensions: " + str(config))
        controlScript = await role.aask(config, MultiDimensionalAcquisitionScript)
        result = await self.client.execute_code(controlScript.script)
        return ExecutionResult(
            status=result['status'],
            outputs=result['outputs'],
            traceback=result.get("traceback")
        )

def create_microscopist(client=None):
    if not client:
        client = create_mock_client()
    microscope = Microscope(client)
    Microscopist = Role.create(
        name="Thomas",
        profile="Microscopist",
        goal="Acquire images from the microscope based on user's requests.",
        constraints=None,
        actions=[microscope.plan, microscope.snap, microscope.move, microscope.multi_dimensional_acquisition],
    )
    return Microscopist()

async def main():
    ms = create_microscopist()
    ms.recv(Message(content="acquire an image every 1 second for 10 seconds", role="User"))
    resp = await ms._react()
    print(resp)
    for res in resp:
        ms.recv(res)
        resp = await ms._react()
        print(resp)
    

if __name__ == "__main__":
    asyncio.run(main())

