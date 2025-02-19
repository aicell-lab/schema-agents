import os
import pytest
import asyncio
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Dict, Union, Tuple, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydantic_ai import RunContext, models, exceptions
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, ToolCallPart
from pydantic_ai.usage import Usage
from pydantic_ai.tools import Tool

from schema_agents.agent import Agent
from schema_agents.reasoning import ReasoningStrategy, ReActConfig

# Load environment variables
load_dotenv()

# Enable real model requests for testing
models.ALLOW_MODEL_REQUESTS = True

# Test Models and Dependencies
class AnalysisResult(BaseModel):
    """Test model for structured agent output"""
    message: str = Field(..., description="Analysis message")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    tags: List[str] = Field(default_factory=list, description="Analysis tags")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")

class ReasoningConfig(BaseModel):
    """Configuration for agent reasoning strategies"""
    type: Union[str, List[str]] = Field(..., description="Reasoning strategy type(s)")
    max_steps: int = Field(3, description="Maximum reasoning steps")
    min_confidence: float = Field(0.8, description="Minimum confidence threshold")
    reflection_rounds: Optional[int] = Field(None, description="Number of reflection rounds")

@dataclass
class TestDependencies:
    """Test dependencies for agent"""
    history: List[str]
    context: Dict[str, str]
    vector_store: Optional[Dict] = None
    tool_calls: Dict[str, int] = dataclasses.field(default_factory=dict)
    
    def __init__(self):
        self.history = []
        self.context = {}
        self.vector_store = {}
        self.tool_calls = {}
    
    async def add_to_history(self, entry: str):
        self.history.append(entry)
    
    async def get_context(self, key: str) -> Optional[str]:
        return self.context.get(key)
    
    async def store_vector(self, key: str, vector: List[float]):
        self.vector_store[key] = vector
    
    async def search_vectors(self, query_vector: List[float], top_k: int = 3) -> List[str]:
        # Simulate vector search
        return list(self.vector_store.keys())[:top_k]
    
    def record_tool_call(self, tool_name: str):
        """Record a tool call."""
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1

# Fixtures
@pytest.fixture
def openai_model():
    """Create an OpenAI model instance for testing"""
    return OpenAIModel(
        'gpt-4o-mini',
        api_key=os.getenv('OPENAI_API_KEY')
    )

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

@pytest.fixture
def test_deps():
    """Create test dependencies"""
    return TestDependencies()

# Test Cases
@pytest.mark.asyncio
async def test_agent_basic_interaction(test_agent, test_deps):
    """Test basic agent interaction with string output"""
    result = await test_agent.run(
        "Hello, how are you?",
        deps=test_deps
    )
    assert isinstance(result.data, str)
    assert len(result.data) > 0

@pytest.mark.asyncio
async def test_agent_structured_output(structured_agent, test_deps):
    """Test agent with structured output"""
    result = await structured_agent.run(
        "Analyze this test message",
        deps=test_deps
    )
    assert isinstance(result.data, AnalysisResult)
    assert 0 <= result.data.confidence <= 1
    assert isinstance(result.data.tags, list)

@pytest.mark.asyncio
async def test_agent_with_tools():
    """Test agent with tool integration"""
    @dataclass
    class ToolDeps:
        counter: int = 0
        memory: Dict[str, str] = None
        
        def __init__(self):
            self.counter = 0
            self.memory = {}
    
    agent = Agent(
        model=OpenAIModel('gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY')),
        name="Tool Agent",
        deps_type=ToolDeps,
        result_type=str
    )
    
    @agent.tool
    async def increment_counter(ctx: RunContext[ToolDeps]) -> str:
        """Increment the counter"""
        ctx.deps.counter += 1
        return f"Counter is now {ctx.deps.counter}"
    
    @agent.tool
    async def store_memory(ctx: RunContext[ToolDeps], key: str, value: str) -> str:
        """Store a value in memory"""
        ctx.deps.memory[key] = value
        return f"Stored {value} under key {key}"
    
    deps = ToolDeps()
    result = await agent.run(
        "Please increment the counter and store the number 42 with key 'answer'",
        deps=deps
    )
    
    assert deps.counter == 1
    assert deps.memory.get('answer') == '42'
    assert isinstance(result.data, str)

@pytest.mark.asyncio
async def test_agent_streaming(test_agent, test_deps):
    """Test agent streaming capability"""
    async with test_agent.run_stream(
        "Give me a long response",
        deps=test_deps
    ) as response:
        chunks = []
        async for chunk, last in response.stream_structured(debounce_by=0.01):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        final_result = await response.get_data()
        assert isinstance(final_result, str)

@pytest.mark.asyncio
async def test_agent_memory_persistence(test_agent, test_deps):
    """Test agent memory/history persistence"""
    # First interaction
    await test_agent.run(
        "Remember the number 42",
        deps=test_deps
    )
    assert len(test_deps.history) > 0
    
    # Second interaction referencing first
    result = await test_agent.run(
        "What number did I ask you to remember?",
        deps=test_deps
    )
    assert "42" in result.data.lower()

@pytest.mark.asyncio
async def test_agent_with_vector_memory(openai_model):
    """Test agent with vector memory integration"""
    agent = Agent(
        model=openai_model,
        name="Vector Memory Agent",
        deps_type=TestDependencies,
        result_type=str
    )
    
    deps = TestDependencies()
    
    # Store some test vectors
    await deps.store_vector("doc1", [0.1, 0.2, 0.3])
    await deps.store_vector("doc2", [0.4, 0.5, 0.6])
    
    @agent.tool
    async def search_knowledge_base(
        ctx: RunContext[TestDependencies],
        query_vector: List[float],
        top_k: int = 3
    ) -> List[str]:
        """Search the vector store for similar documents"""
        return await ctx.deps.search_vectors(query_vector, top_k)
    
    result = await agent.run(
        "Search for similar documents",
        deps=deps
    )
    assert isinstance(result.data, str)

@pytest.mark.asyncio
async def test_agent_with_dynamic_tools(openai_model):
    """Test agent with dynamically attached tools"""
    agent = Agent(
        model=openai_model,
        name="Dynamic Tool Agent",
        deps_type=TestDependencies,
        result_type=str
    )
    
    async def dynamic_tool(x: int) -> int:
        return x * 2
    
    async def another_tool(text: str) -> str:
        return text.upper()
    
    result = await agent.run(
        "Process some data",
        deps=TestDependencies(),
        tools=[dynamic_tool, another_tool]
    )
    assert isinstance(result.data, str)

@pytest.mark.asyncio
async def test_agent_with_schema_tools():
    """Test agent with schema tools that use Field for parameter descriptions"""
    from pydantic import Field
    from typing import Optional

    class HelloMockModel(models.Model):
        def __init__(self):
            self._model_name = "mock-model"
            self._has_tool_return = False
        
        @property
        def model_name(self) -> str:
            return self._model_name
        
        @property
        def system(self) -> str:
            return "You are a mock model for testing schema tools."
        
        async def request(
            self,
            message_history: List[ModelMessage],
            model_settings: Dict[str, Any] | None = None,
            request_parameters: Dict[str, Any] | None = None,
        ) -> Tuple[ModelResponse, Usage]:
            """Return a tool call response for say_hello"""
            # Get the last request
            last_message = message_history[-1]
            if isinstance(last_message, ModelRequest):
                # Check if we have a tool return in any of the messages
                for msg in message_history:
                    if any(isinstance(p, ToolReturnPart) for p in msg.parts):
                        self._has_tool_return = True
                        break
                
                if not self._has_tool_return:
                    # First return the tool call
                    return ModelResponse(
                        parts=[ToolCallPart(
                            tool_name='say_hello',
                            args={"name": "Bob"},
                            tool_call_id='test-call'
                        )],
                        model_name=self.model_name
                    ), Usage()
                else:
                    # Then return the final result
                    return ModelResponse(
                        parts=[TextPart(content="Hello Bob!")],
                        model_name=self.model_name
                    ), Usage()
            return ModelResponse(
                parts=[TextPart(content="I don't understand")],
                model_name=self.model_name
            ), Usage()

    class CalculateMockModel(models.Model):
        def __init__(self):
            self._model_name = "mock-model"
            self._has_tool_return = False
        
        @property
        def model_name(self) -> str:
            return self._model_name
        
        @property
        def system(self) -> str:
            return "You are a mock model for testing schema tools."
        
        async def request(
            self,
            message_history: List[ModelMessage],
            model_settings: Dict[str, Any] | None = None,
            request_parameters: Dict[str, Any] | None = None,
        ) -> Tuple[ModelResponse, Usage]:
            """Return a tool call response for calculate"""
            # Get the last request
            last_message = message_history[-1]
            if isinstance(last_message, ModelRequest):
                # Check if we have a tool return in any of the messages
                for msg in message_history:
                    if any(isinstance(p, ToolReturnPart) for p in msg.parts):
                        self._has_tool_return = True
                        break
                
                if not self._has_tool_return:
                    # First return the tool call
                    return ModelResponse(
                        parts=[ToolCallPart(
                            tool_name='calculate',
                            args={"x": 5, "y": 3, "operation": "add"},
                            tool_call_id='test-call'
                        )],
                        model_name=self.model_name
                    ), Usage()
                else:
                    # Then return the final result
                    return ModelResponse(
                        parts=[TextPart(content="5 + 3 = 8")],
                        model_name=self.model_name
                    ), Usage()
            return ModelResponse(
                parts=[TextPart(content="I don't understand")],
                model_name=self.model_name
            ), Usage()

    # Test with default parameters
    agent1 = Agent(
        model=HelloMockModel(),
        name="Schema Tool Agent",
        deps_type=TestDependencies,
        result_type=str,
        retries=3  # Increase retries for testing
    )

    @agent1.schema_tool
    async def say_hello(
        name: Optional[str] = Field(default="alice", description="Name of the person to greet")
    ) -> str:
        """Say hello to someone"""
        return f"Hello {name}!"
    
    # Verify tool definition schema
    tool_def = await say_hello.__tool__.prepare_tool_def(None)
    assert tool_def is not None
    params = tool_def.parameters_json_schema
    assert params["properties"]["name"]["description"] == "Name of the person to greet"
    assert params["properties"]["name"]["default"] == "alice"

    result1 = await agent1.run(
        "Say hello",
        deps=TestDependencies(),
        usage_limits=None  # Disable usage limits for testing
    )
    assert result1.data == "Hello Bob!"

    # Test calculation with custom parameters
    agent2 = Agent(
        model=CalculateMockModel(),
        name="Schema Tool Agent",
        deps_type=TestDependencies,
        result_type=str,
        retries=3  # Increase retries for testing
    )

    @agent2.schema_tool
    async def calculate(
        x: int = Field(..., description="First number"),
        y: int = Field(default=10, description="Second number, defaults to 10"),
        operation: str = Field(default="add", description="Operation to perform: add/subtract")
    ) -> str:
        """Perform a calculation"""
        if operation == "add":
            return f"{x} + {y} = {x + y}"
        elif operation == "subtract":
            return f"{x} - {y} = {x - y}"
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # Verify tool definition schema
    tool_def = await calculate.__tool__.prepare_tool_def(None)
    assert tool_def is not None
    params = tool_def.parameters_json_schema
    assert params["properties"]["x"]["description"] == "First number"
    assert "default" not in params["properties"]["x"]
    assert params["properties"]["y"]["description"] == "Second number, defaults to 10"
    assert params["properties"]["y"]["default"] == 10
    assert params["properties"]["operation"]["description"] == "Operation to perform: add/subtract"
    assert params["properties"]["operation"]["default"] == "add"

    result2 = await agent2.run(
        "Calculate 5 plus 3",
        deps=TestDependencies(),
        usage_limits=None  # Disable usage limits for testing
    )
    assert result2.data == "5 + 3 = 8"

@pytest.mark.asyncio
async def test_agent_error_handling():
    """Test various error handling scenarios"""
    # Create a mock model that returns invalid responses
    class ErrorMockModel(models.Model):
        def __init__(self, response_type: str = "invalid"):
            self.response_type = response_type
            self._model_name = "mock-model"
        
        @property
        def model_name(self) -> str:
            return self._model_name
        
        @property
        def system(self) -> str:
            return "You are a mock model for testing."
        
        async def request(
            self,
            message_history: List[ModelMessage],
            model_settings: Dict[str, Any] | None = None,
            request_parameters: Dict[str, Any] | None = None,
        ) -> Tuple[ModelResponse, Usage]:
            """Return an invalid response based on response_type"""
            if self.response_type == "invalid":
                # Return an invalid response that will fail validation
                raise exceptions.UserError("Invalid response format")
            elif self.response_type == "error":
                # Raise an error
                raise ValueError("Model error")
            else:
                # Return a valid response
                return ModelResponse(
                    parts=[TextPart(content='{"message": "Test", "confidence": 0.9, "tags": []}')],
                    model_name=self.model_name
                ), Usage()
    
    # Test invalid model response
    agent = Agent(
        model=ErrorMockModel("invalid"),
        name="Error Agent",
        deps_type=TestDependencies,
        result_type=AnalysisResult,
        retries=3  # Increase retries for testing
    )
    
    with pytest.raises(exceptions.UserError):
        await agent.run(
            "Return a string instead of an AnalysisResult",
            deps=TestDependencies(),
            usage_limits=None  # Disable usage limits for testing
        )
    
    # Test missing dependencies
    with pytest.raises(exceptions.UserError):
        await agent.run("Hello")
    
    # Test model error
    agent_with_error = Agent(
        model=ErrorMockModel("error"),
        name="Error Agent",
        deps_type=TestDependencies,
        result_type=AnalysisResult,
        retries=3  # Increase retries for testing
    )
    
    with pytest.raises(exceptions.UserError):
        await agent_with_error.run(
            "This should raise an error",
            deps=TestDependencies(),
            usage_limits=None  # Disable usage limits for testing
        )

@pytest.mark.asyncio
async def test_agent_override(openai_model):
    """Test agent model override functionality"""
    agent = Agent(
        model=openai_model,
        name="Override Agent",
        deps_type=TestDependencies,
        result_type=str
    )
    
    # Create a new OpenAI model with different settings
    new_model = OpenAIModel(
        'gpt-4o-mini',
        api_key=os.getenv('OPENAI_API_KEY'),
        system_prompt_role="Test Role"
    )
    
    # Test with original model
    result1 = await agent.run(
        "Hello",
        deps=TestDependencies()
    )
    
    # Test with overridden model
    result2 = await agent.run(
        "Hello",
        deps=TestDependencies(),
        model=new_model
    )
    
    assert isinstance(result1.data, str)
    assert isinstance(result2.data, str)

@pytest.mark.asyncio
async def test_agent_with_real_openai_reasoning(openai_model):
    """Test reasoning strategy with real OpenAI model."""
    # Create agent with ReAct reasoning
    agent = Agent(
        model=openai_model,
        name="Real OpenAI ReAct Agent",
        deps_type=TestDependencies,
        result_type=str,
        role="Problem Solver",
        goal="Solve problems step by step using ReAct reasoning",
        reasoning_strategy=ReasoningStrategy(
            type="react",
            react_config=ReActConfig(
                max_loops=10,
                min_confidence=0.8
            )
        )
    )
    
    # Add tools
    async def search_wikipedia(ctx: RunContext[TestDependencies], query: str) -> str:
        """Search Wikipedia for information."""
        ctx.deps.record_tool_call('search_wikipedia')
        return f"Wikipedia results for: {query}"

    async def calculate(ctx: RunContext[TestDependencies], expression: str) -> float:
        """Calculate a mathematical expression."""
        ctx.deps.record_tool_call('calculate')
        return eval(expression)
    
    tools = [search_wikipedia, calculate]
    deps = TestDependencies()
    
    # Test with a simple math problem
    result = await agent.run(
        "What is 25 * 48? Then find information about the number in Wikipedia.",
        deps=deps,
        tools=tools
    )
    
    # Verify result
    assert isinstance(result.data, str)
    assert "1200" in result.data
    assert "Wikipedia" in result.data
    
    # Verify tool calls
    assert deps.tool_calls.get('calculate', 0) >= 1
    assert deps.tool_calls.get('search_wikipedia', 0) >= 1
