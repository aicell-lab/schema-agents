"""Test agent registration with Hypha."""
import pytest
from schema_agents import Agent
from pydantic_ai import RunContext
from hypha_rpc import connect_to_server
from .test_agent import AnalysisResult
from .test_utils import TestDependencies
from pydantic_ai.models.openai import OpenAIModel
from schema_agents.reasoning import ReasoningStrategy, ReActConfig

# Global variables for tracking callback execution
callback_called = False
callback_error = None
callback_chunks = []

def reset_callback_state():
    """Reset global callback state."""
    global callback_called, callback_error, callback_chunks
    callback_called = False
    callback_error = None
    callback_chunks = []

@pytest.fixture
def test_agent(openai_model):
    """Create a basic test agent with OpenAI model"""
    return Agent(
        model=openai_model,
        name="Test Agent",
        deps_type=TestDependencies,
        result_type=str,
        role="Test Assistant",
        goal="Help with testing the agent implementation",
        backstory="You are an AI assistant helping with testing."
    )

@pytest.fixture
def test_deps():
    """Create test dependencies"""
    return TestDependencies()

@pytest.fixture
def structured_agent(openai_model):
    """Create an agent with structured output using OpenAI model"""
    return Agent(
        model=openai_model,
        name="Structured Agent",
        deps_type=TestDependencies,
        result_type=AnalysisResult,
        role="Analysis Assistant",
        goal="Provide structured analysis results",
        backstory="You are an AI assistant that provides structured analysis."
    )

@pytest.mark.asyncio
async def test_agent_registration(test_agent, test_deps, hypha_server_url):
    """Test registering an agent as a Hypha service."""
    async with connect_to_server({
        "server_url": hypha_server_url
    }) as server:
        # Register the agent
        service_info = await test_agent.register(server, "test-agent")
        service = await server.get_service(service_info.id)

        # Verify service registration
        assert service.id.endswith(":test-agent")  # Service ID includes workspace prefix
        assert service.name == test_agent.name
        assert "Agent service for" in service.description
        assert "Role: Test Assistant" in service.docs
        assert "Goal: Help with testing" in service.docs

        # Test using the service
        result = await service.run("Hello, how are you?", deps=test_deps.model_dump() if hasattr(test_deps, 'model_dump') else test_deps.__dict__)
        assert isinstance(result, str)
        assert len(result) > 0

        # Test streaming
        chunks = []
        async def callback(chunk):
            chunks.append(chunk)

        try:
            result = await service.run("Tell me about yourself", callback, deps=test_deps.model_dump() if hasattr(test_deps, 'model_dump') else test_deps.__dict__)
            assert isinstance(result, str)
            assert len(result) > 0
            assert len(chunks) > 0
        except Exception as e:
            if "Event loop is closed" in str(e):
                pytest.skip("Skipping streaming test due to event loop issues")
            else:
                raise e

@pytest.mark.asyncio
async def test_agent_registration_with_tools(openai_model, hypha_server_url):
    """Test registering an agent with tools as a Hypha service."""
    async with connect_to_server({
        "server_url": hypha_server_url
    }) as server:
        # Create agent with tools
        agent = Agent(
            model=openai_model,
            name="Tool Agent",
            deps_type=TestDependencies,
            result_type=str,
            role="Tool Assistant",
            goal="Help with tool testing"
        )

        @agent.tool
        async def add_numbers(ctx: RunContext[TestDependencies], x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        @agent.tool
        async def multiply_numbers(ctx: RunContext[TestDependencies], x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        # Register agent
        service_info = await agent.register(server, "tool-agent")
        service = await server.get_service(service_info.id)
        
        # Verify tool metadata
        assert any(tool["name"] == "add_numbers" for tool in service.tools)
        assert any(tool["name"] == "multiply_numbers" for tool in service.tools)
        
        # Test using tools through service
        deps = TestDependencies()
        result = await service.run("What is 5 + 3?", deps=deps.model_dump() if hasattr(deps, 'model_dump') else deps.__dict__)
        assert "8" in result.lower()

@pytest.mark.asyncio
async def test_agent_registration_with_structured_output(structured_agent, hypha_server_url):
    """Test registering an agent with structured output as a Hypha service."""
    async with connect_to_server({
        "server_url": hypha_server_url
    }) as server:
        # Register agent
        service_info = await structured_agent.register(server, "structured-agent")
        service = await server.get_service(service_info.id)

        # Verify result type in metadata
        assert service.result_type == "AnalysisResult"
        
        # Test using service with structured output
        deps = TestDependencies()
        result = await service.run("Analyze this test message", deps=deps.model_dump() if hasattr(deps, 'model_dump') else deps.__dict__)
        assert isinstance(result, dict)  # Result will be a dict since it's coming through Hypha
        result_obj = AnalysisResult.model_validate(result)  # Convert back to AnalysisResult
        assert 0 <= result_obj.confidence <= 1

@pytest.mark.asyncio
async def test_agent_registration_with_reasoning(openai_model, hypha_server_url):
    """Test registering an agent with reasoning strategy as a Hypha service."""
    async with connect_to_server({
        "server_url": hypha_server_url
    }) as server:
        # Create agent with reasoning strategy
        agent = Agent(
            model=openai_model,
            name="Reasoning Agent",
            deps_type=TestDependencies,
            result_type=str,
            role="Problem Solver",
            goal="Solve problems step by step",
            reasoning_strategy=ReasoningStrategy(
                type="react",
                react_config=ReActConfig(
                    max_loops=5,
                    min_confidence=0.8
                )
            )
        )
        
        # Register agent
        service_info = await agent.register(server, "reasoning-agent")
        service = await server.get_service(service_info.id)
        
        # Verify reasoning strategy in metadata
        assert service.reasoning_strategy["type"] == "react"
        assert service.reasoning_strategy["react_config"]["max_loops"] == 5
        
        # Test using service with reasoning
        deps = TestDependencies()
        result = await service.run("What is 2 + 2?", deps=deps.model_dump() if hasattr(deps, 'model_dump') else deps.__dict__)
        assert "4" in result.lower()

@pytest.mark.asyncio
async def test_agent_registration_client_usage(test_agent, hypha_server_url):
    """Test using a registered agent from a different client."""
    # Reset callback state
    reset_callback_state()
    
    async with connect_to_server({
        "server_url": hypha_server_url
    }) as server:
        # Register agent
        service_id = "client-test-agent"
        service_info = await test_agent.register(server, service_id)
        service = await server.get_service(service_info.id)
        assert service is not None
        assert service.id.endswith(":client-test-agent")

        # Connect as a different client
        client = await connect_to_server({"server_url": hypha_server_url})
        try:
            # Get service
            service = await client.get_service(service.id)  # Use the full service ID
            assert service is not None
            assert service.id.endswith(":client-test-agent")

            # Test running the service
            test_prompt = "Hello, can you help me with testing?"
            deps = TestDependencies()
            result = await service.run(test_prompt, deps=deps.model_dump() if hasattr(deps, 'model_dump') else deps.__dict__)
            assert isinstance(result, str)
            assert len(result) > 0

            # Test streaming
            async def callback(chunk: dict):
                global callback_called, callback_error, callback_chunks
                try:
                    callback_called = True
                    assert isinstance(chunk, dict)
                    assert "type" in chunk
                    assert "content" in chunk
                    assert "model" in chunk
                    assert isinstance(chunk["type"], str)
                    assert isinstance(chunk["content"], str)
                    assert len(chunk["content"]) > 0
                    callback_chunks.append(chunk)
                except Exception as e:
                    callback_error = e

            stream_result = await service.run(test_prompt, callback=callback, deps=deps.model_dump() if hasattr(deps, 'model_dump') else deps.__dict__)
            assert isinstance(stream_result, str)
            assert len(stream_result) > 0
            
            # Verify callback execution
            assert callback_called, "Callback was not called"
            assert callback_error is None, f"Callback error: {callback_error}"
            assert len(callback_chunks) > 0, "No chunks were received"
        finally:
            await client.disconnect() 
