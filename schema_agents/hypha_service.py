import asyncio
import re
import inspect
from pydantic import BaseModel
from hypha_rpc import connect_to_server
from schema_agents import Role
from schema_agents.role import create_session_context
from schema_agents.schema import StreamEvent
from schema_agents.utils.common import EventBus
from schema_agents.utils.jsonschema_pydantic import json_schema_to_pydantic_model
from schema_agents import schema_tool

def create_tool_name(svc_id, tool_id=""):
    text = f"{svc_id}_{tool_id}"
    text = text.replace("-", " ").replace("_", " ").replace(".", " ")
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+', text)
    return ''.join(word if word.istitle() else word.capitalize() for word in words)

def tool_factory(svc_id, tool_id, tool_func, schema):
    schema["title"] = create_tool_name(svc_id, tool_id)
    schema["description"] = tool_func.__doc__
    input_model = json_schema_to_pydantic_model(schema, ref_template="$defs")
    
    func_name = create_tool_name(svc_id, tool_id)
    func_doc = input_model.__doc__ or tool_func.__doc__

    async def wrapper(*args, **kwargs):
        def convert_basemodel(obj):
            if isinstance(obj, BaseModel):
                return obj.model_dump(mode="json")
            elif isinstance(obj, dict):
                return {k: convert_basemodel(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_basemodel(item) for item in obj]
            return obj

        converted_args = [convert_basemodel(arg) for arg in args]
        converted_kwargs = {k: convert_basemodel(v) for k, v in kwargs.items()}
        
        result = tool_func(*converted_args, **converted_kwargs)
        if inspect.isawaitable(result):
            result = await result
        # convert results to dict
        return convert_basemodel(result)

    wrapper.__name__ = func_name
    wrapper.__doc__ = func_doc
    
    return schema_tool(wrapper, input_model=input_model)

async def aask(question, agent_config, streaming_callback=None):
    """Ask a question."""
    agent = Role(**agent_config)
    event_bus = EventBus("test")
    if streaming_callback:
        async def callback(response: StreamEvent):
            await streaming_callback(response.model_dump(mode="json"))
        event_bus.on("stream", callback)
    async with create_session_context(
        event_bus=event_bus
    ):
        return await agent.aask(question)

def extract_tools_from_service(service):
    """A utility function to extract functions nested in a service."""
    if isinstance(service, dict):
        for name, value in service.items():
            yield from extract_tools_from_service(value)
    elif isinstance(service, (list, tuple)):
        yield from extract_tools_from_service(service)
    elif callable(service):
        yield service

def normalize_service_name(text):
    """Normalize the service name to be used in the tool usage prompt."""
    text = text.replace("-", " ").replace("_", " ").replace(".", " ")
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+', text)
    return ''.join(word if word.istitle() else word.capitalize() for word in words)

async def register_agent_service(server):
    """Register a service with the server."""

    async def acall(question, agent_config, tools=None, services=None, streaming_callback=None, **kwargs):
        """Ask a question."""
        agent = Role(**agent_config)
        event_bus = EventBus("test")
        if streaming_callback:
            async def callback(response: StreamEvent):
                await streaming_callback(response.model_dump(mode="json"))
            event_bus.on("stream", callback)

        if services:
            tools = []
            service_prompts = []
            for service in services:
                if isinstance(service, str):
                    service = await server.get_service(service)
                
                svc_id = normalize_service_name(service['id'].split(":")[-1])
                ts = list(extract_tools_from_service(service))
                for t in ts:
                    schema = t.__schema__
                    tools.append(tool_factory(svc_id, t.__name__, t, schema["parameters"]))

                svcd = service['description'].replace('\n', ' ')
                service_prompts.append(f" - {svc_id}*: {svcd}\n")
        
        tool_usage_prompt = "Tool usage guidelines (* represent the prefix of a tool group):\n" + "\n".join(service_prompts)

        async with create_session_context(
            event_bus=event_bus
        ):
            return await agent.acall(question, tools=tools, tool_usage_prompt=tool_usage_prompt, **kwargs)

    svc = await server.register_service({
        "name": "Schema Agents",
        "id": "schema-agents",
        "config": {
            "visibility": "public"
        },
        "aask": aask,
        "acall": acall,
    })
    return svc

async def main():
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
    })
    svc = await register_agent_service(server)
    print(f"Agent service registered: {svc.id}")
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())