import pytest
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel

from schema_agents.chain_reasoning import ChainReasoning, ChainLink, ThoughtChain
from schema_agents.schema_reasoning import Thought, ThoughtType, Action, ActionType, Observation
from pydantic_ai import RunContext, models, result, exceptions
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, ToolCallPart
from pydantic_ai.usage import Usage

class ChainMockModel(models.Model):
    """Mock model for testing chain reasoning."""
    def __init__(self):
        self._model_name = "mock-chain"
        self._step = 0
        self._branch_count = 0
        self._last_observation = None
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def system(self) -> str:
        return "You are a mock model for testing chain reasoning."
    
    async def request(
        self,
        message_history: List[ModelMessage],
        model_settings: Dict[str, Any] | None = None,
        request_parameters: Dict[str, Any] | None = None,
    ) -> Tuple[ModelResponse, Usage]:
        """Simulate chain reasoning responses."""
        # Get the last request
        last_message = message_history[-1]
        if not isinstance(last_message, ModelRequest):
            return ModelResponse(
                parts=[TextPart(content="I don't understand")],
                model_name=self.model_name
            ), Usage()
        
        content = last_message.parts[0].content
        
        # Handle next thought generation
        if "Generate the next thought" in content:
            self._step += 1
            is_branch = "alternative approaches" in content
            
            if self._step > 5:  # Conclude after 5 steps
                thought = Thought(
                    type=ThoughtType.CONCLUDE,
                    content=f"Final conclusion after {self._step} steps",
                    confidence=0.9,
                    supporting_facts=[f"Fact {i}" for i in range(2)],
                    next_action=None
                )
            else:
                thought = Thought(
                    type=ThoughtType.ANALYZE,
                    content=f"{'Alternative ' if is_branch else ''}Step {self._step}",
                    confidence=0.8 if not is_branch else 0.7,
                    supporting_facts=[f"Fact {i}" for i in range(2)],
                    next_action="calculate" if self._step % 2 == 0 else None
                )
            
            return ModelResponse(
                parts=[TextPart(content=str(thought))],
                model_name=self.model_name
            ), Usage()
        
        # Handle action generation
        if "Generate action" in content:
            action = Action(
                type=ActionType.TOOL_CALL,
                tool_name="calculate",
                arguments={"expression": f"{self._step} + {self._step}"},
                purpose="Perform calculation",
                expected_outcome="Numeric result",
                fallback="Use approximation"
            )
            return ModelResponse(
                parts=[TextPart(content=str(action))],
                model_name=self.model_name
            ), Usage()
        
        # Handle step evaluation
        if "Evaluate" in content:
            score = 0.9 - (0.1 * (self._step % 3))  # Cycle through scores
            return ModelResponse(
                parts=[TextPart(content=str(score))],
                model_name=self.model_name
            ), Usage()
        
        # Handle branching decision
        if "explore alternative approaches" in content.lower():
            self._branch_count += 1
            should_branch = self._branch_count <= 2  # Only branch twice
            return ModelResponse(
                parts=[TextPart(content=str(should_branch))],
                model_name=self.model_name
            ), Usage()
        
        return ModelResponse(
            parts=[TextPart(content="Default response")],
            model_name=self.model_name
        ), Usage()

@pytest.fixture
def chain_reasoner():
    """Create a test chain reasoning instance."""
    model = ChainMockModel()
    tools = []  # Add mock tools if needed
    return ChainReasoning(
        model=model,
        tools=tools,
        max_steps=10,
        min_confidence=0.7,
        branching_factor=2,
        max_branches=3
    )

@pytest.mark.asyncio
async def test_chain_basic_execution(chain_reasoner):
    """Test basic chain reasoning execution."""
    class TestDeps:
        pass
    
    run_context = RunContext(deps=TestDeps())
    
    # Execute reasoning
    result = await chain_reasoner.execute(
        "Solve this problem step by step",
        run_context
    )
    
    # Verify result format and content
    assert isinstance(result, str)
    assert "Step" in result
    assert "Final conclusion" in result

@pytest.mark.asyncio
async def test_chain_streaming(chain_reasoner):
    """Test chain reasoning with streaming."""
    class TestDeps:
        pass
    
    run_context = RunContext(deps=TestDeps())
    
    # Execute with streaming
    chunks = []
    async for chunk in await chain_reasoner.execute(
        "Solve this problem step by step",
        run_context,
        stream=True
    ):
        chunks.append(chunk)
    
    # Verify streaming output
    assert len(chunks) > 0
    assert any("Step" in chunk for chunk in chunks)
    assert any("Final answer" in chunk for chunk in chunks)

@pytest.mark.asyncio
async def test_chain_branching(chain_reasoner):
    """Test chain branching functionality."""
    class TestDeps:
        pass
    
    run_context = RunContext(deps=TestDeps())
    
    # Modify branching settings
    chain_reasoner.branching_factor = 1
    result1 = await chain_reasoner.execute(
        "Solve this problem step by step",
        run_context
    )
    
    chain_reasoner.branching_factor = 2
    result2 = await chain_reasoner.execute(
        "Solve this problem step by step",
        run_context
    )
    
    # Results should be different due to different branching
    assert result1 != result2
    assert "Alternative" in result2

@pytest.mark.asyncio
async def test_chain_max_steps(chain_reasoner):
    """Test maximum steps limit."""
    class TestDeps:
        pass
    
    run_context = RunContext(deps=TestDeps())
    
    # Set very low step limit
    chain_reasoner.max_steps = 2
    result = await chain_reasoner.execute(
        "Solve this problem step by step",
        run_context
    )
    
    # Should get shorter result
    steps = result.count("Step")
    assert steps <= 2

@pytest.mark.asyncio
async def test_chain_confidence_threshold(chain_reasoner):
    """Test confidence threshold."""
    class TestDeps:
        pass
    
    run_context = RunContext(deps=TestDeps())
    
    # Set high confidence threshold
    chain_reasoner.min_confidence = 0.95
    result = await chain_reasoner.execute(
        "Solve this problem step by step",
        run_context
    )
    
    # Should get shorter result due to high threshold
    assert len(result.split("\n")) < 5

def test_thought_chain_operations():
    """Test ThoughtChain data structure operations."""
    chain = ThoughtChain()
    
    # Create initial thought and action
    thought1 = Thought(
        type=ThoughtType.ANALYZE,
        content="Initial thought",
        confidence=0.9,
        supporting_facts=["Fact 1"],
        next_action="calculate"
    )
    action1 = Action(
        type=ActionType.TOOL_CALL,
        tool_name="calculate",
        arguments={"x": 1, "y": 2},
        purpose="Test calculation",
        expected_outcome="Sum",
        fallback=None
    )
    
    # Add initial link
    link1_idx = chain.add_link(
        thought=thought1,
        action=action1,
        score=0.9
    )
    
    # Add a branch
    thought2 = Thought(
        type=ThoughtType.ANALYZE,
        content="Alternative approach",
        confidence=0.8,
        supporting_facts=["Fact 2"],
        next_action=None
    )
    link2_idx = chain.add_link(
        thought=thought2,
        score=0.8,
        branch_from=link1_idx
    )
    
    # Test chain structure
    assert len(chain.links) == 2
    assert link1_idx in chain.branches
    assert link2_idx in chain.branches[link1_idx]
    
    # Test branch points
    branch_points = chain.get_branch_points()
    assert link1_idx in branch_points
    
    # Test best branch
    best_branch = chain.get_best_branch(link1_idx)
    assert best_branch == link2_idx 