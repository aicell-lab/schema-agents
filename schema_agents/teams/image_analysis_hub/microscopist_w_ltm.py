import asyncio
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message, MemoryChunk
from schema_agents.memory.long_term_memory import LongTermMemory
from schema_agents.tools.code_interpreter import create_mock_client
from .schemas import (FunctionMemory, ErrorMemory)

class MicroscopeControlRequirements(BaseModel):
    """Requirements for controlling the microscope and acquire images."""
    path: str = Field(default="", description="save images path")
    timeout: float = Field(default=0.0, description="timeout")
    query: str = Field(default="", description="user's original request")
    plan: str = Field(default="", description="plan for control microscope and acquiring images")
    experiences: List[MemoryChunk] = Field(default=[], description="experiences of making errors")

class MultiDimensionalAcquisitionScript(BaseModel):
    """Python script for simple and complex multi-dimensional acquisition.
    In the script, you can use the following functions to control the microscope:
    - `microscope_move({'x': 0.0, 'y': 0.0, 'z': 0.0})` # x, y, z are in microns
    - `microscope_snap({'path': './images', 'exposure': 0.0})` # path is the path to save the image, exposure is in seconds
    """
    script: str = Field(default="", description="Script for acquiring multi-dimensional images")
    explanation: str = Field(default="", description="Brief explanation for the script")
    timeout: float = Field(default=0.0, description="a reasonable timeout for executing the script")

class ExecutionResult(BaseModel):
    """Result of executing a Python script."""
    status: str = Field(description="Status of executing the script")
    outputs: List[Dict[str, Any]] = Field(default=[], description="Outputs of executing the script")
    traceback: Optional[str] = Field(default=None, description="Traceback of executing the script")



def create_long_term_memory():
    memory = LongTermMemory()
    role_id = 'bio'
    memory.recover_memory(role_id)
    memory.clean()
    
    function_move = FunctionMemory(function_name='microscope_move', code="""def microscope_move(position):
        print(f"===> Moving to: {position}")""", lang='python', args=['position'])
    function_snap = FunctionMemory(function_name='microscope_snap', code="""def microscope_snap(config):
        print(f"===> Snapped an image with exposure {config['exposure']} and saved to: { config['path']}")""", lang='python', args=['config'])

    error = ErrorMemory(error='Microscope move can not be obtained for less than 5nm', solution='Make sure each movement is larger than 5nm')

    error_memo = MemoryChunk(index='Error made for microscope_move function', content=error, category='error')
    memory.add(error_memo)
    new_memory = MemoryChunk(index='microscope move python function',content=function_move, category='function')
    memory.add(new_memory)
    new_memory = MemoryChunk(index='microscope snap python function',content=function_snap, category='function')
    memory.add(new_memory)
    memories = memory.recover_memory(role_id)
    return memory


class Microscope():
    def __init__(self, client):
        self.client = client
        self.initialized = False

    async def plan(self, query: str=None, role: Role=None) -> MicroscopeControlRequirements:
        """Make a plan for image acquisition tasks."""
        return await role.aask(query, MicroscopeControlRequirements)
        
    async def multi_dimensional_acquisition(self, config: MicroscopeControlRequirements=None, role: Role=None) -> ExecutionResult:
        """Perform image acquisition by using Python script."""
        if not self.initialized:
            memories = role.long_term_memory.retrieve("microscope related functions", filter={"category": "function"})
            for memory in memories:
                script = memory.content.code
                await self.client.executeScript({"script": script})
            self.initialized = True

        experiences = role.long_term_memory.retrieve("microscope related function", filter={"category": "error"})
        config.experiences = experiences
        print("Acquiring images in multiple dimensions: " + str(config))
        controlScript = await role.aask(config, MultiDimensionalAcquisitionScript)
        result = await self.client.executeScript({"script": controlScript.script, "timeout": controlScript.timeout})
        if result['status'] != 'ok':
            new_experience = await role.aask('summarize the error experience', ErrorMemory)
            error_memo = MemoryChunk(index='Error made for microscope_move function', content=new_experience, category='error')
            role.long_term_memory.add(error_memo)
            
        return ExecutionResult(
            status=result['status'],
            outputs=result['outputs'],
            traceback=result.get("traceback")
        )

def create_microscopist_with_ltm(client=None):
    if not client:
        client = create_mock_client()

    microscope = Microscope(client)
    Microscopist = Role.create(
        name="Thomas",
        profile="Microscopist",
        goal="Acquire images from the microscope based on user's requests.",
        constraints=None,
        actions=[microscope.plan, microscope.multi_dimensional_acquisition],
    )
    
    return Microscopist


async def main():
    client = create_mock_client()
    microscope = Microscope(client)

    Microscopist = Role.create(
        name="Thomas",
        profile="Microscopist",
        goal="Acquire images from the microscope based on user's requests.",
        constraints=None,
        actions=[microscope.plan, microscope.multi_dimensional_acquisition],
        long_term_memory=create_long_term_memory(),
    )
    ms = Microscopist()

    ms.recv(Message(content="acquire image every 2nm along x, y in a 2x2um square, gradually increase exposure time from 0.1 to 2.0s", role="User"))
    resp = await ms._react()
    print(resp)
    for res in resp:
        ms.recv(res)
        resp = await ms._react()
        print(resp)
    

    ms.recv(Message(content="acquire an image and save to /tmp/img.png", role="User"))
    resp = await ms._react()
    print(resp)
    for res in resp:
        ms.recv(res)
        resp = await ms._react()
        print(resp)

    ms.recv(Message(content="acquire an image every 1 second for 10 seconds", role="User"))
    resp = await ms._react()
    print(resp)
    for res in resp:
        ms.recv(res)
        resp = await ms._react()
        print(resp)
    
    ms.long_term_memory.clean()
    assert ms.long_term_memory.is_initialized is False

if __name__ == "__main__":
    asyncio.run(main())
    # create_memory_storage()

