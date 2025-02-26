import pytest
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel

from schema_agents.agent import Agent
from schema_agents.memory import SemanticMemory
from schema_agents.tree_reasoning import TreeReasoning
from schema_agents.chain_reasoning import ChainReasoning
from schema_agents.visualization import visualize_memory_state
from schema_agents.reasoning_factory import ReasoningConfig, create_reasoning_strategy
from schema_agents.schema_reasoning import (
    ReasoningState, Thought, Action, Observation,
    ThoughtType, ActionType, MemoryType
)
from pydantic_ai import RunContext, models, result, exceptions
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, ToolCallPart
from pydantic_ai.usage import Usage

class IntegrationTestModel(models.Model):
    """Mock model for integration testing."""
    def __init__(self):
        self._model_name = "mock-integration"
        self._step = 0
        self._memory = SemanticMemory()
        self._last_observation = None
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def system(self) -> str:
        return "You are a mock model for integration testing."
    
    async def request(
        self,
        message_history: List[ModelMessage],
        model_settings: Dict[str, Any] | None = None,
        request_parameters: Dict[str, Any] | None = None,
    ) -> Tuple[ModelResponse, Usage]:
        """Simulate model responses for integration testing."""
        # Get the last request
        last_message = message_history[-1]
        if not isinstance(last_message, ModelRequest):
            return ModelResponse(
                parts=[TextPart(content="I don't understand")],
                model_name=self.model_name
            ), Usage()
        
        content = last_message.parts[0].content
        
        # Handle different request types based on content
        if "solve" in content.lower() and "math" in content.lower():
            self._step += 1
            if self._step == 1:
                # Initial analysis
                thought = Thought(
                    type=ThoughtType.ANALYZE,
                    content="Let's break down the math problem",
                    confidence=0.9,
                    supporting_facts=["Math requires step by step solution"],
                    next_action="calculate"
                )
                return ModelResponse(
                    parts=[TextPart(content=str(thought))],
                    model_name=self.model_name
                ), Usage()
            elif self._step == 2:
                # Calculation action
                action = Action(
                    type=ActionType.TOOL_CALL,
                    tool_name="calculate",
                    arguments={"expression": "2 + 2"},
                    purpose="Basic calculation",
                    expected_outcome="Sum",
                    fallback=None
                )
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name="calculate",
                        args={"expression": "2 + 2"},
                        tool_call_id="calc-1"
                    )],
                    model_name=self.model_name
                ), Usage()
            else:
                # Conclusion
                return ModelResponse(
                    parts=[TextPart(content="Final Answer: 2 + 2 = 4")],
                    model_name=self.model_name
                ), Usage()
        
        elif "search" in content.lower():
            # Simulate search behavior
            self._step += 1
            return ModelResponse(
                parts=[ToolCallPart(
                    tool_name="search",
                    args={"query": content},
                    tool_call_id="search-1"
                )],
                model_name=self.model_name
            ), Usage()
        
        # Default response for evaluation
        return ModelResponse(
            parts=[TextPart(content=str(0.9))],
            model_name=self.model_name
        ), Usage()

class TestDependencies:
    """Test dependencies with memory and tool tracking."""
    def __init__(self):
        self.memory = SemanticMemory()
        self.tool_calls = {}
        self.history = []
    
    def record_tool_call(self, tool_name: str):
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
    
    async def add_to_history(self, entry: str):
        self.history.append(entry)

@pytest.fixture
def test_model():
    """Create test model instance."""
    return IntegrationTestModel()

@pytest.fixture
def test_deps():
    """Create test dependencies instance."""
    return TestDependencies()

@pytest.mark.asyncio
async def test_full_reasoning_pipeline(test_model, test_deps):
    """Test complete reasoning pipeline with all components."""
    # Create agent with all reasoning strategies
    agent = Agent(
        model=test_model,
        name="Integration Agent",
        deps_type=TestDependencies,
        result_type=str,
        role="Problem Solver",
        goal="Solve problems using multiple reasoning strategies",
        reasoning_strategy=ReasoningConfig(
            strategy=["react", "tree", "chain"],
            max_steps=5,
            max_depth=3,
            beam_width=2,
            branching_factor=2
        )
    )
    
    # Add tools
    @agent.tool
    async def calculate(ctx: RunContext[TestDependencies], expression: str) -> float:
        """Calculate a mathematical expression."""
        ctx.deps.record_tool_call('calculate')
        return eval(expression)
    
    @agent.tool
    async def search(ctx: RunContext[TestDependencies], query: str) -> str:
        """Search for information."""
        ctx.deps.record_tool_call('search')
        return f"Results for: {query}"
    
    # Test with a problem that requires multiple reasoning steps
    result = await agent.run(
        "Solve this math problem: 2 + 2, then search for information about the number 4",
        deps=test_deps
    )
    
    # Verify result
    assert isinstance(result.data, str)
    assert "4" in result.data
    assert test_deps.tool_calls.get('calculate', 0) > 0
    assert test_deps.tool_calls.get('search', 0) > 0
    assert len(test_deps.history) > 0

@pytest.mark.asyncio
async def test_memory_integration(test_model, test_deps):
    """Test memory integration with reasoning."""
    # Add some initial memories
    test_deps.memory.add_entry(
        MemoryType.FACT,
        "Addition is a basic mathematical operation",
        metadata={"source": "math knowledge"}
    )
    test_deps.memory.add_entry(
        MemoryType.OBSERVATION,
        "Previous calculations were accurate",
        metadata={"confidence": 0.9}
    )
    
    # Create agent
    agent = Agent(
        model=test_model,
        name="Memory Agent",
        deps_type=TestDependencies,
        result_type=str,
        role="Problem Solver",
        goal="Solve problems using memory"
    )
    
    # Add tools
    @agent.tool
    async def calculate(ctx: RunContext[TestDependencies], expression: str) -> float:
        """Calculate a mathematical expression."""
        ctx.deps.record_tool_call('calculate')
        result = eval(expression)
        # Store result in memory
        ctx.deps.memory.add_entry(
            MemoryType.OBSERVATION,
            f"Calculated {expression} = {result}",
            metadata={"expression": expression, "result": result}
        )
        return result
    
    # Run agent
    result = await agent.run(
        "What is 2 + 2?",
        deps=test_deps
    )
    
    # Verify memory integration
    assert isinstance(result.data, str)
    assert "4" in result.data
    
    # Search memory for calculation
    memories = test_deps.memory.search("calculation")
    assert len(memories) > 0
    assert any("2 + 2" in str(entry.content) for entry, _ in memories)

@pytest.mark.asyncio
async def test_visualization_integration(test_model, test_deps):
    """Test visualization integration with reasoning."""
    # Create agent with visualization
    agent = Agent(
        model=test_model,
        name="Visual Agent",
        deps_type=TestDependencies,
        result_type=str,
        role="Problem Solver",
        goal="Solve problems with visual feedback"
    )
    
    # Add tools
    @agent.tool
    async def calculate(ctx: RunContext[TestDependencies], expression: str) -> float:
        """Calculate a mathematical expression."""
        ctx.deps.record_tool_call('calculate')
        return eval(expression)
    
    # Run agent and collect state
    result = await agent.run(
        "Calculate 2 + 2",
        deps=test_deps
    )
    
    # Generate visualizations
    viz = visualize_memory_state(agent.state, format="all")
    
    # Verify visualization outputs
    assert "memory_graph" in viz
    assert "memory_timeline" in viz
    assert "state_diagram" in viz
    assert "confidence_chart" in viz
    
    # Verify visualization content
    assert "Task" in viz["memory_graph"]
    assert "Calculate" in viz["memory_graph"]
    assert "FACT" in viz["memory_timeline"]
    assert "stateDiagram" in viz["state_diagram"]
    assert "Confidence" in viz["confidence_chart"]

@pytest.mark.asyncio
async def test_reasoning_strategy_composition(test_model, test_deps):
    """Test composition of different reasoning strategies."""
    # Create agent with multiple reasoning strategies
    agent = Agent(
        model=test_model,
        name="Multi-Strategy Agent",
        deps_type=TestDependencies,
        result_type=str,
        role="Problem Solver",
        goal="Solve problems using multiple strategies",
        reasoning_strategy=ReasoningConfig(
            strategy=["react", "tree", "chain"],
            max_steps=5,
            max_depth=3,
            beam_width=2,
            branching_factor=2
        )
    )
    
    # Add tools
    @agent.tool
    async def calculate(ctx: RunContext[TestDependencies], expression: str) -> float:
        """Calculate a mathematical expression."""
        ctx.deps.record_tool_call('calculate')
        return eval(expression)
    
    @agent.tool
    async def search(ctx: RunContext[TestDependencies], query: str) -> str:
        """Search for information."""
        ctx.deps.record_tool_call('search')
        return f"Results for: {query}"
    
    # Test with streaming to observe strategy transitions
    chunks = []
    async with agent.run_stream(
        "First calculate 2 + 2, then search about the number 4, finally make a conclusion",
        deps=test_deps
    ) as response:
        async for chunk in response._stream_response:
            chunks.append(chunk)
    
    # Verify strategy composition
    assert len(chunks) > 0
    assert test_deps.tool_calls.get('calculate', 0) > 0
    assert test_deps.tool_calls.get('search', 0) > 0
    
    # Verify reasoning transitions in history
    history_text = " ".join(test_deps.history)
    assert "react" in history_text.lower()
    assert "tree" in history_text.lower()
    assert "chain" in history_text.lower() 