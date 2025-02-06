import asyncio
from hypha_rpc import connect_to_server
from pydantic import BaseModel, Field
from hypha_rpc.utils.schema import schema_function
from schema_agents.hypha_service import register_agent_service

class UserInfo(BaseModel):
    """User information."""
    name: str = Field(..., description="Name of the user")
    email: str = Field(..., description="Email of the user")
    age: int = Field(..., description="Age of the user")
    address: str = Field(..., description="Address of the user")

@schema_function
def register_user(user_info: UserInfo) -> str:
    """Register a new user."""
    return f"User {user_info.name} registered"

async def run_chatbot():
    server_url = "https://hypha.aicell.io"

    async def streaming_callback(message: dict):
        if message["type"] == "function_call":
            if message["status"] == "in_progress":
                print(message["arguments"], end="", flush=True)
            else:
                print(f'\nGenerating {message["name"]} ({message["status"]}): {message["arguments"]}', flush=True)
        elif message["type"] == "text":
            print(message["content"], end="", flush=True)

    # Connect to server
    async with connect_to_server(server_url=server_url) as server:

        agent_svc = await register_agent_service(server)
        # Get the agent service
        agent = await server.get_service(agent_svc.id)
        
        # Configure the agent
        agent_config = {
            "name": "Alice",
            "profile": "BioImage Analyst",
            "goal": "A are a helpful agent",
            "model": "gpt-4o-mini",
            # "model": "qwen2.5-coder",
            # "client_config": {
            #     "base_url": 'https://hypha-ollama.scilifelab-2-dev.sys.kth.se/v1',
            #     "api_key": 'ollama',
            # },
            "stream": True
        }
        
        # Ask the agent
        # answer = await agent.aask(
        #     "Hi, say hello to me! Then tell me an interesting story",
        #     agent_config=agent_config,
        #     streaming_callback=streaming_callback
        # )
        # print(answer)

        svc = await server.register_service(
            {
                "id": "user-service",
                "name": "User Service",
                "description": "A service for managing users",
                "register_user": register_user,
                "docs": "This service specializes in managing users and their information."
            }
        )


        # user_service = await server.get_service(svc.id)
        # services = await server.search_services(query="user", limit=3)
        # assert len(services) >= 1

        # Call the agent
        answer = await agent.acall(
            "Hi, say hello to me! help me register a new user with name 'John Doe', email 'john.doe@example.com', age 30, and address '123 Main St, Anytown, USA'",
            agent_config=agent_config,
            # services=[svc.id for svc in services],
            services=[svc.id],
            streaming_callback=streaming_callback
        )
        print(answer)

        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(run_chatbot())