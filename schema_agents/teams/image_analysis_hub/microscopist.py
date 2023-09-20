import asyncio
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.tools.code_interpreter import create_mock_client
from schema_agents.memory.longterm_memory import LongTermMemory
from schema_agents.memory.memory_storage import MemoryStorage


class MicroscopeControlRequirements(BaseModel):
    """Requirements for controlling the microscope and acquire images."""
    path: str = Field(default="", description="save images path")
    timeout: float = Field(default=0.0, description="timeout")
    query: str = Field(default="", description="user's original request")
    plan: str = Field(default="", description="plan for control microscope and acquiring images")

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

INIT_SCRIPT = """
def microscope_move(position):
    print(f"===> Moving to: {position}")

def microscope_snap(config):
    print(f"===> Snapped an image with exposure {config['exposure']} and saved to: { config['path']}")
"""

def test_idea_message():
    idea = 'Write a cli snake game'
    role_id = 'UTUser1(Product Manager)'
    message = Message(role='BOSS', content=INIT_SCRIPT, cause_by=MicroscopeControlRequirements)

    memory_storage: MemoryStorage = MemoryStorage()
    messages = memory_storage.recover_memory(role_id)
    memory_storage.add(message)
    messages = memory_storage.recover_memory(role_id)
    # assert len(messages) == 0

    memory_storage.add(message)
    assert memory_storage.is_initialized is True

    sim_idea = 'Write a game of cli snake'
    sim_message = Message(role='BOSS', content=sim_idea, cause_by=MicroscopeControlRequirements)
    new_messages = memory_storage.search(sim_message)
    assert len(new_messages) == 0   # similar, return []

    memory_storage.add(sim_message)
    search_message = memory_storage.search(sim_message)
    assert len(search_message) == 1
    
    new_idea = 'Write a 2048 web game'
    new_message = Message(role='BOSS', content=new_idea, cause_by=MicroscopeControlRequirements)
    new_messages = memory_storage.search(new_message)
    assert new_messages[0].content == message.content

    memory_storage.clean()
    assert memory_storage.is_initialized is False

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
            await self.client.executeScript({"script": INIT_SCRIPT})
            self.initialized = True
        print("Acquiring images in multiple dimensions: " + str(config))
        controlScript = await role.aask(config, MultiDimensionalAcquisitionScript)
        result = await self.client.executeScript({"script": controlScript.script, "timeout": controlScript.timeout})
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
        actions=[microscope.multi_dimensional_acquisition],
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
    

if __name__ == "__main__":
    # asyncio.run(main())
    test_idea_message()

