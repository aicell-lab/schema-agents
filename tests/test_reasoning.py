import pytest
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from pydantic_ai import RunContext, models, result, exceptions
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, ToolCallPart
from pydantic_ai.usage import Usage
from schema_agents.agent import Agent
from schema_agents.reasoning import ReasoningStrategy, ReActConfig

# Test dependencies
@dataclass
class TestDeps:
    history: List[str]
    tool_calls: Dict[str, int]
    
    def __init__(self):
        self.history = []
        self.tool_calls = {}
    
    async def add_to_history(self, entry: str):
        self.history.append(entry)
    
    def record_tool_call(self, tool_name: str):
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1

# Test tools
async def search_wikipedia(ctx: RunContext[TestDeps], query: str) -> str:
    """Search Wikipedia for information."""
    ctx.deps.record_tool_call('search_wikipedia')
    return f"Wikipedia results for: {query}"

async def calculate(ctx: RunContext[TestDeps], expression: str) -> float:
    """Calculate a mathematical expression."""
    ctx.deps.record_tool_call('calculate')
    return eval(expression)

class ReActMockModel(models.Model):
    """Mock model that simulates ReAct pattern."""
    def __init__(self):
        self._model_name = "mock-react"
        self._step = 0
        self._last_observation = None
        self._request_count = 0
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def system(self) -> str:
        return "You are a mock model for testing ReAct reasoning."
    
    async def request(
        self,
        message_history: List[ModelMessage],
        model_settings: Dict[str, Any] | None = None,
        model_request_parameters: Dict[str, Any] | None = None,
    ) -> Tuple[ModelResponse, Usage]:
        """Simulate ReAct pattern responses."""
        self._request_count += 1
        
        # Get the last request
        last_message = message_history[-1]
        if not isinstance(last_message, ModelRequest):
            return ModelResponse(
                parts=[TextPart(content="I don't understand")],
                model_name=self.model_name
            ), Usage()
        
        # Extract current question and observation
        content = last_message.parts[0].content
        if "Current query: " in content:
            question = content.split("Current query: ")[1].split("\n")[0]
            observation = content.split("Previous observation: ")[1].split("\n")[0]
            if observation == "None":
                observation = None
            self._last_observation = observation
        else:
            question = content
            observation = None
        
        # Simulate ReAct pattern
        if "25 * 48" in question:
            if self._step == 0:
                # First calculate
                self._step += 1
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name='calculate',
                        args={"expression": "25 * 48"},
                        tool_call_id='calc-1'
                    )],
                    model_name=self.model_name
                ), Usage()
            elif self._step == 1 and self._last_observation:
                # Then search Wikipedia
                self._step += 1
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name='search_wikipedia',
                        args={"query": "1200"},
                        tool_call_id='wiki-1'
                    )],
                    model_name=self.model_name
                ), Usage()
            elif self._step == 2 and self._last_observation:
                # Finally give answer
                self._step = 0
                return ModelResponse(
                    parts=[TextPart(content=f"Final Answer: The result of 25 * 48 is 1200. {self._last_observation}")],
                    model_name=self.model_name
                ), Usage()
        elif "complex calculation" in question.lower():
            # Simulate a more complex calculation requiring multiple steps
            if self._step < 3:  # Will make 3 calculation calls
                self._step += 1
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name='calculate',
                        args={"expression": f"{self._step} + {self._step}"},
                        tool_call_id=f'calc-{self._step}'
                    )],
                    model_name=self.model_name
                ), Usage()
            else:
                # Reset step and return final answer
                self._step = 0
                return ModelResponse(
                    parts=[TextPart(content=f"Final Answer: Completed {self._request_count} steps of calculations")],
                    model_name=self.model_name
                ), Usage()
        elif "2 + 2" in question:
            # Simple calculation
            return ModelResponse(
                parts=[TextPart(content="Final Answer: 2 + 2 = 4")],
                model_name=self.model_name
            ), Usage()
        
        # Default response
        return ModelResponse(
            parts=[TextPart(content="Let me help you with that step by step.")],
            model_name=self.model_name
        ), Usage()

@pytest.mark.asyncio
async def test_react_reasoning():
    """Test ReAct reasoning strategy."""
    # Create agent with ReAct reasoning
    agent = Agent(
        model=ReActMockModel(),
        name="ReAct Agent",
        deps_type=TestDeps,
        result_type=str,
        role="Problem Solver",
        goal="Solve problems step by step using ReAct reasoning",
        reasoning_strategy=ReasoningStrategy(
            type="react",
            react_config=ReActConfig(
                max_loops=5,
                min_confidence=0.8
            )
        )
    )
    
    # Add tools
    tools = [search_wikipedia, calculate]
    deps = TestDeps()
    
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
    assert deps.tool_calls['calculate'] == 1
    assert deps.tool_calls['search_wikipedia'] == 1
    
    # Test with a complex calculation
    deps = TestDeps()
    result = await agent.run(
        "Perform a complex calculation with multiple steps.",
        deps=deps,
        tools=tools
    )
    
    # Verify multiple calculation steps
    assert deps.tool_calls['calculate'] == 3
    assert "Completed" in result.data

@pytest.mark.asyncio
async def test_react_with_max_loops():
    """Test ReAct reasoning with max loops limit."""
    agent = Agent(
        model=ReActMockModel(),
        name="ReAct Agent",
        deps_type=TestDeps,
        result_type=str,
        role="Problem Solver",
        goal="Solve problems step by step using ReAct reasoning",
        reasoning_strategy=ReasoningStrategy(
            type="react",
            react_config=ReActConfig(
                max_loops=2,  # Set a low max loops
                min_confidence=0.8
            )
        )
    )
    
    # Add tools
    tools = [search_wikipedia, calculate]
    deps = TestDeps()
    
    # Test with a complex problem that requires multiple steps
    with pytest.raises(Exception) as exc_info:
        await agent.run(
            "Perform a complex calculation that requires many steps.",
            deps=deps,
            tools=tools
        )
    
    assert "exceeded maximum loop count" in str(exc_info.value).lower()
    # Verify we made some tool calls before hitting the limit
    assert deps.tool_calls.get('calculate', 0) > 0

@pytest.mark.asyncio
async def test_react_with_confidence():
    """Test ReAct reasoning with confidence threshold."""
    agent = Agent(
        model=ReActMockModel(),
        name="ReAct Agent",
        deps_type=TestDeps,
        result_type=str,
        role="Problem Solver",
        goal="Solve problems step by step using ReAct reasoning",
        reasoning_strategy=ReasoningStrategy(
            type="react",
            react_config=ReActConfig(
                max_loops=5,
                min_confidence=0.99  # Set a very high confidence threshold
            )
        )
    )
    
    # Add tools
    tools = [search_wikipedia, calculate]
    deps = TestDeps()
    
    # Test with a simple problem
    result = await agent.run(
        "What is 2 + 2?",
        deps=deps,
        tools=tools
    )
    
    # The calculation should be confident enough to pass
    assert "4" in result.data
    # Verify no tool calls were needed for this simple case
    assert len(deps.tool_calls) == 0
