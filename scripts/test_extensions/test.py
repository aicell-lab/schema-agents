import extensions
from pydantic import BaseModel, Field
from typing import Optional
from schema_agents.role import Role
from schema_agents.schema import Message

class ResponseStep(BaseModel):
    """Response step"""
    name : str = Field(description = "Step Name")
    details : Optional[dict] = Field(None, description = "Step Details")

class RichResponse(BaseModel):
    """Rich response with text and intermediate steps"""
    text : str = Field(..., description = "Response text")
    steps : Optional[list[ResponseStep]] = Field(None, description = "Intermediate steps")

def create_assistants():
    
    async def respond(
            query : str, role : Role = None
    ) -> RichResponse:
        """Answers the user's question directory or retrieve relevant information, or create a Python Script to get information about details of models."""
        steps = []
        inputs = (query)
        builtin_extensions = extensions.get_builtin_extensions()
        tools = []
        for extension in builtin_extensions:
            tool = await extensions.extension_to_tool(extension)
            tools.append(tool)
        
        class ThoughtsSchema(BaseModel):
            """Details about the thoughts"""
            reasoning: str = Field(..., description="reasoning and constructive self-criticism; make it short and concise in less than 20 words")
        
        response, metadata = await role.acall(inputs,
                                              tools,
                                              return_metadata=True,
                                              thoughts_schema=ThoughtsSchema,
                                              max_loop_count = 10)
        result_steps = metadata['steps']
        for idx, step_list in enumerate(result_steps):
            steps.append(
                ResponseStep(
                    name = f"step-{idx}",
                    details = {"details" : convert_to_dict(step_list)}
                )
            )
        return RichResponse(text = response, steps = steps)
    
    assistant = Role(
        instructions = "You are the assistant, a helpful agent for helping the user",
        actions = [respond],
        model = "gpt-4-0125-preview",
    )
    event_bus = assistant.get_event_bus()
    event_bus.register_default_events()
    all_extensions = [{"name" : ext.name, "description" : ext.description} for ext in builtin_extensions]
    return [{"name" : "assistant", "agent" : assistant, "extensions" : all_extensions}]

if __name__ == "__main__":
    assistant = create_assistants()[0]
    print(assistant)