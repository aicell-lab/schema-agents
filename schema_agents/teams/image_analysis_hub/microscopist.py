import asyncio
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message, MemoryChunk
from schema_agents.memory.long_term_memory import LongTermMemory
from schema_agents.tools.code_interpreter import create_mock_client
from schemas import (FunctionMemory, ExperienceMemory)

class MicroscopeControlRequirements(BaseModel):
    """Requirements for controlling the microscope and acquire images."""
    path: str = Field(default="", description="save images path")
    timeout: float = Field(default=0.0, description="timeout")
    query: str = Field(default="", description="user's original request")
    plan: str = Field(default="", description="plan for control microscope and acquiring images")
    

class MultiDimensionalAcquisitionScript(BaseModel):
    """Python script for simple and complex multi-dimensional acquisition. Use the provided 'Experiences' to generate the script."""
    script: str = Field(default="", description="Script for acquiring multi-dimensional images")
    explanation: str = Field(default="", description="Brief explanation for the script")
    timeout: float = Field(default=0.0, description="a reasonable timeout for executing the script")

class ExecutionResult(BaseModel):
    """Result of executing a Python script."""
    status: str = Field(description="Status of executing the script")
    outputs: List[Dict[str, Any]] = Field(default=[], description="Outputs of executing the script")
    traceback: Optional[str] = Field(default=None, description="Traceback of executing the script")


class FunctionDescription(BaseModel):
    """Description of a function."""
    name: str = Field(description="Name of the function")
    docstring: str = Field(description="Docstring of the function")
    args: List[str] = Field(default=[], description="Arguments of the function with type annotation")
    
    
class ScriptGenerationContext(BaseModel):
    """The context for generating the script."""
    registed_functions: List[FunctionDescription] = Field(default=[], description="A list of registed functions which can be used directly in the script.")
    experience_memory: List[ExperienceMemory] = Field(default=[], description="A list of experience memories to help create the script and avoid common mistakes.")

def create_long_term_memory():
    memory = LongTermMemory()
    role_id = 'bio'
    memory.recover_memory(role_id)
    memory.clean()
    
    function_move = FunctionMemory(function_name='microscope_move', code="""def microscope_move(position):
        print(f"===> Moving to: {position}")""", lang='python', args=['position:Tuple[float,float,float]'], docstring='Move the microscope to the given position.')
    function_snap = FunctionMemory(function_name='microscope_snap', code="""def microscope_snap(config):
        print(f"===> Snapped an image with exposure {config['exposure']} and saved to: { config['path']}")""", lang='python', args=['config:Dict[str,Any]'], 
        docstring='Snap an image with the given configuration. The input config should contain the exposure time (key=exposure) and the path (key=path) to save the image.')

    exp = ExperienceMemory(summary='Microscope move restriction', keypoints='Make sure each movement on microscope is larger than 5nm.')

    exp_memo = MemoryChunk(index='Experience for running microscope_move function', content=exp, category='experience')
    memory.add(exp_memo)
    fun1_memo = MemoryChunk(index='microscope move python function',content=function_move, category='function')
    memory.add(fun1_memo)
    fun2_memo = MemoryChunk(index='microscope snap python function',content=function_snap, category='function')
    memory.add(fun2_memo)
    memories = memory.recover_memory(role_id)
    return memory


class Microscope():
    def __init__(self, client):
        self.client = client
        self.initialized = True

    async def plan(self, query: str=None, role: Role=None) -> MicroscopeControlRequirements:
        """Make a plan for image acquisition tasks."""
        return await role.aask(query, MicroscopeControlRequirements)
        
    async def multi_dimensional_acquisition(self, config: MicroscopeControlRequirements=None, role: Role=None) -> ExecutionResult:
        """Perform image acquisition by using Python script."""
        
        function_memories = role.long_term_memory.retrieve("microscope related functions", filter={"category": "function"})
        function_list = []
        for memory in function_memories:
            script = memory.content.code
            function_list.append(FunctionDescription(name=memory.content.function_name, docstring=memory.content.docstring, args=memory.content.args))
            await self.client.executeScript({"script": script})
        experiences = role.long_term_memory.retrieve("microscope related function", filter={"category": "experience"})
        exp_list = []
        for exp in experiences:
            exp_list.append(exp.content)
        context = ScriptGenerationContext(registed_functions=function_list, experience_memory=exp_list)
        print("Acquiring images in multiple dimensions: " + str(config))

        inputs = [context, config, """Make sure each movement on the microscope stage is larger than 5nm."""]
        controlScript = await role.aask(inputs, MultiDimensionalAcquisitionScript)
        result = await self.client.executeScript({"script": controlScript.script, "timeout": controlScript.timeout})

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

