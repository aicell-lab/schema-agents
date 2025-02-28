import pytest
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from pydantic_ai import RunContext, models, result, exceptions
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, ToolCallPart
from pydantic_ai.usage import Usage
from schema_agents.agent import Agent
from schema_agents.reasoning import ReasoningStrategy, StructuredReasoningConfig, StructuredReasoningResult

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

class StructuredMockModel(models.Model):
    """Mock model that simulates Structured Reasoning pattern."""
    def __init__(self):
        self._model_name = "mock-structured"
        self._step = 0
        self._last_observation = None
        self._request_count = 0
        self._current_plan = None
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def system(self) -> str:
        return "You are a mock model for testing Structured Reasoning."
    
    async def request(
        self,
        message_history: List[ModelMessage],
        model_settings: Dict[str, Any] | None = None,
        model_request_parameters: Dict[str, Any] | None = None,
    ) -> Tuple[ModelResponse, Usage]:
        """Simulate Structured Reasoning pattern responses."""
        self._request_count += 1
        
        # Get the last request
        last_message = message_history[-1]
        
        # Handle dictionary messages
        if isinstance(last_message, dict):
            content = last_message.get("content", "")
        elif isinstance(last_message, ModelRequest):
            if not last_message.parts:
                return ModelResponse(
                    parts=[TextPart(content="I don't understand")],
                    model_name=self.model_name
                ), Usage()
            content = last_message.parts[0].content
        else:
            return ModelResponse(
                parts=[TextPart(content="I don't understand")],
                model_name=self.model_name
            ), Usage()
        
        # Check if this is a summarization request
        if "Please summarize the following reasoning steps:" in content:
            response = ModelResponse(
                parts=[TextPart(content="Summary of reasoning: The agent performed calculations and searched for information.")],
                model_name=self.model_name
            )
            return response, Usage()
        
        # Simulate Structured Reasoning pattern
        if "25 * 48" in content:
            if self._step == 0:
                # First create a plan
                self._step += 1
                plan = ["Calculate 25 * 48", "Search for information about the result"]
                self._current_plan = plan
                
                result = StructuredReasoningResult(
                    thought="I need to calculate 25 * 48 first",
                    plan=plan,
                    action="print(25 * 48)",
                    message="I'll calculate 25 * 48 for you"
                )
                
                # Create a ModelResponse with the result data
                response = ModelResponse(
                    parts=[TextPart(content="Structured reasoning result")],
                    model_name=self.model_name
                )
                # Add the data attribute manually
                response.data = result.model_dump()
                
                return response, Usage()
            elif self._step == 1:
                # Then search Wikipedia
                self._step += 1
                
                result = StructuredReasoningResult(
                    thought="Now I'll search for information about 1200",
                    plan=self._current_plan,
                    action=None,
                    message="The result is 1200. Let me search for more information."
                )
                
                # Create a ModelResponse with the result data
                response = ModelResponse(
                    parts=[TextPart(content="Structured reasoning result")],
                    model_name=self.model_name
                )
                # Add the data attribute manually
                response.data = result.model_dump()
                
                return response, Usage()
            elif self._step == 2:
                # Final answer
                self._step = 0
                
                result = StructuredReasoningResult(
                    thought="I've completed the calculation and search",
                    plan=self._current_plan,
                    action=None,
                    message="The result of 25 * 48 is 1200. I've completed the calculation."
                )
                
                # Create a ModelResponse with the result data
                response = ModelResponse(
                    parts=[TextPart(content="Structured reasoning result")],
                    model_name=self.model_name
                )
                # Add the data attribute manually
                response.data = result.model_dump()
                
                return response, Usage()
        elif "complex calculation" in content.lower() or "complex calculation that requires many steps" in content.lower():
            # Simulate a more complex calculation requiring multiple steps
            if self._step < 3:  # Will make 3 calculation calls
                self._step += 1
                
                # Create a plan if this is the first step
                plan = self._current_plan
                if self._step == 1:
                    plan = ["Step 1: Calculate first value", "Step 2: Calculate second value", "Step 3: Calculate final value"]
                    self._current_plan = plan
                
                result = StructuredReasoningResult(
                    thought=f"Performing calculation step {self._step}",
                    plan=plan,
                    action=f"print({self._step} + {self._step})",
                    message=f"Performing calculation step {self._step}"
                )
                
                # Create a ModelResponse with the result data
                response = ModelResponse(
                    parts=[TextPart(content="Structured reasoning result")],
                    model_name=self.model_name
                )
                # Add the data attribute manually
                response.data = result.model_dump()
                
                return response, Usage()
            else:
                # Reset step and return final answer
                self._step = 0
                
                result = StructuredReasoningResult(
                    thought="I've completed all calculation steps",
                    plan=self._current_plan,
                    action=None,
                    message=f"Completed {self._request_count} steps of calculations"
                )
                
                # Create a ModelResponse with the result data
                response = ModelResponse(
                    parts=[TextPart(content="Structured reasoning result")],
                    model_name=self.model_name
                )
                # Add the data attribute manually
                response.data = result.model_dump()
                
                return response, Usage()
        elif "2 + 2" in content:
            # Simple calculation
            result = StructuredReasoningResult(
                thought="This is a simple calculation",
                plan=["Calculate 2 + 2"],
                action=None,
                message="2 + 2 = 4"
            )
            
            # Create a ModelResponse with the result data
            response = ModelResponse(
                parts=[TextPart(content="Structured reasoning result")],
                model_name=self.model_name
            )
            # Add the data attribute manually
            response.data = result.model_dump()
            
            return response, Usage()
        
        # Default response
        result = StructuredReasoningResult(
            thought="I'm not sure how to proceed",
            plan=None,
            action=None,
            message="I'm not sure how to help with that"
        )
        
        # Create a ModelResponse with the result data
        response = ModelResponse(
            parts=[TextPart(content="Structured reasoning result")],
            model_name=self.model_name
        )
        # Add the data attribute manually
        response.data = result.model_dump()
        
        return response, Usage()

@pytest.mark.asyncio
async def test_structured_reasoning():
    """Test Structured Reasoning strategy."""
    # Create agent with Structured Reasoning
    agent = Agent(
        model=StructuredMockModel(),
        name="Structured Agent",
        deps_type=TestDeps,
        result_type=str,
        role="Problem Solver",
        goal="Solve problems step by step using Structured Reasoning",
        reasoning_strategy=ReasoningStrategy(
            type="structured",
            structured_config=StructuredReasoningConfig(
                max_steps=5,
                memory_enabled=True,
                summarize_memory=True
            )
        )
    )
    
    # Add tools
    tools = [search_wikipedia, calculate]
    deps = TestDeps()
    
    # Test with a simple math problem
    result = await agent.run(
        "What is 25 * 48?",
        deps=deps,
        tools=tools
    )
    
    # Verify result
    assert isinstance(result.data, str)
    assert "1200" in result.data
    
    # Test with a complex calculation
    deps = TestDeps()
    result = await agent.run(
        "Perform a complex calculation with multiple steps.",
        deps=deps,
        tools=tools
    )
    
    # Verify result contains the summary
    assert "Summary of reasoning" in result.data

@pytest.mark.asyncio
async def test_structured_with_max_steps():
    """Test Structured Reasoning with max steps limit."""
    agent = Agent(
        model=StructuredMockModel(),
        name="Structured Agent",
        deps_type=TestDeps,
        result_type=str,
        role="Problem Solver",
        goal="Solve problems step by step using Structured Reasoning",
        reasoning_strategy=ReasoningStrategy(
            type="structured",
            structured_config=StructuredReasoningConfig(
                max_steps=1,  # Set a low max steps
                memory_enabled=True,
                summarize_memory=True
            )
        )
    )
    
    # Add tools
    tools = [search_wikipedia, calculate]
    deps = TestDeps()
    
    # Test with a complex problem that requires multiple steps
    result = await agent.run(
        "Perform a complex calculation that requires many steps.",
        deps=deps,
        tools=tools
    )
    
    # Verify result contains the summary
    assert "Summary of reasoning" in result.data

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_structured_reasoning())
    asyncio.run(test_structured_with_max_steps()) 