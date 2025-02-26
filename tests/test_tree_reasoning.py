import pytest
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel

from schema_agents.tree_reasoning import TreeReasoning, ThoughtTree, ThoughtNode
from schema_agents.schema_reasoning import Thought, ThoughtType, Action, Observation
from pydantic_ai import RunContext, models, result, exceptions
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, ToolCallPart
from pydantic_ai.usage import Usage

class TreeMockModel(models.Model):
    """Mock model for testing tree reasoning."""
    def __init__(self):
        self._model_name = "mock-tree"
        self._step = 0
        self._branch = 0
        self._depth = 0
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def system(self) -> str:
        return "You are a mock model for testing tree reasoning."
    
    async def request(
        self,
        message_history: List[ModelMessage],
        model_settings: Dict[str, Any] | None = None,
        request_parameters: Dict[str, Any] | None = None,
    ) -> Tuple[ModelResponse, Usage]:
        """Simulate tree reasoning responses."""
        # Get the last request
        last_message = message_history[-1]
        if not isinstance(last_message, ModelRequest):
            return ModelResponse(
                parts=[TextPart(content="I don't understand")],
                model_name=self.model_name
            ), Usage()
        
        content = last_message.parts[0].content
        
        # Handle thought generation
        if "Generate" in content and "thoughts" in content:
            thoughts = []
            for i in range(3):  # Generate 3 thoughts
                thought_type = ThoughtType.ANALYZE if self._depth < 2 else ThoughtType.CONCLUDE
                thoughts.append(Thought(
                    type=thought_type,
                    content=f"Branch {self._branch} Thought {i} at depth {self._depth}",
                    confidence=0.8 - (0.1 * i),  # Decreasing confidence
                    supporting_facts=[f"Fact {j}" for j in range(2)],
                    next_action="calculate" if i == 0 else None
                ))
            self._branch += 1
            return ModelResponse(
                parts=[TextPart(content=str(thoughts))],
                model_name=self.model_name
            ), Usage()
        
        # Handle thought evaluation
        if "Evaluate" in content:
            self._step += 1
            score = 0.9 - (0.1 * (self._step % 3))  # Cycle through different scores
            return ModelResponse(
                parts=[TextPart(content=str(score))],
                model_name=self.model_name
            ), Usage()
        
        # Handle terminal check
        if "terminal" in content.lower():
            self._depth += 1
            is_terminal = self._depth >= 3  # Terminal at depth 3
            return ModelResponse(
                parts=[TextPart(content=str(is_terminal))],
                model_name=self.model_name
            ), Usage()
        
        return ModelResponse(
            parts=[TextPart(content="Default response")],
            model_name=self.model_name
        ), Usage()

@pytest.fixture
def tree_reasoner():
    """Create a test tree reasoning instance."""
    model = TreeMockModel()
    tools = []  # Add mock tools if needed
    return TreeReasoning(
        model=model,
        tools=tools,
        max_depth=3,
        beam_width=2,
        min_score=0.5,
        max_branches=3
    )

@pytest.mark.asyncio
async def test_tree_basic_execution(tree_reasoner):
    """Test basic tree reasoning execution."""
    # Create run context
    class TestDeps:
        pass
    
    run_context = RunContext(deps=TestDeps())
    
    # Execute reasoning
    result = await tree_reasoner.execute(
        "Solve this problem step by step",
        run_context
    )
    
    # Verify result format and content
    assert isinstance(result, str)
    assert "Step" in result
    assert "Branch" in result
    assert "Thought" in result

@pytest.mark.asyncio
async def test_tree_streaming(tree_reasoner):
    """Test tree reasoning with streaming."""
    class TestDeps:
        pass
    
    run_context = RunContext(deps=TestDeps())
    
    # Execute with streaming
    chunks = []
    async for chunk in await tree_reasoner.execute(
        "Solve this problem step by step",
        run_context,
        stream=True
    ):
        chunks.append(chunk)
    
    # Verify streaming output
    assert len(chunks) > 0
    assert any("Generated" in chunk for chunk in chunks)
    assert any("Final answer" in chunk for chunk in chunks)

@pytest.mark.asyncio
async def test_tree_beam_search(tree_reasoner):
    """Test beam search functionality."""
    class TestDeps:
        pass
    
    run_context = RunContext(deps=TestDeps())
    
    # Modify beam width
    tree_reasoner.beam_width = 1
    result1 = await tree_reasoner.execute(
        "Solve this problem step by step",
        run_context
    )
    
    tree_reasoner.beam_width = 2
    result2 = await tree_reasoner.execute(
        "Solve this problem step by step",
        run_context
    )
    
    # Results should be different due to different beam widths
    assert result1 != result2

@pytest.mark.asyncio
async def test_tree_depth_limit(tree_reasoner):
    """Test maximum depth limit."""
    class TestDeps:
        pass
    
    run_context = RunContext(deps=TestDeps())
    
    # Set very low depth limit
    tree_reasoner.max_depth = 1
    result = await tree_reasoner.execute(
        "Solve this problem step by step",
        run_context
    )
    
    # Should still get a result, but with fewer steps
    steps = result.count("Step")
    assert steps <= 2  # One for root + one level

@pytest.mark.asyncio
async def test_tree_branching_limit(tree_reasoner):
    """Test maximum branching limit."""
    class TestDeps:
        pass
    
    run_context = RunContext(deps=TestDeps())
    
    # Set very low branching limit
    tree_reasoner.max_branches = 1
    result = await tree_reasoner.execute(
        "Solve this problem step by step",
        run_context
    )
    
    # Should see limited branching in result
    assert result.count("Branch") <= 1

@pytest.mark.asyncio
async def test_tree_score_threshold(tree_reasoner):
    """Test minimum score threshold."""
    class TestDeps:
        pass
    
    run_context = RunContext(deps=TestDeps())
    
    # Set high score threshold
    tree_reasoner.min_score = 0.95
    result = await tree_reasoner.execute(
        "Solve this problem step by step",
        run_context
    )
    
    # Should get shorter result due to high threshold
    assert len(result.split("\n")) < 5  # Fewer steps due to high threshold

def test_thought_tree_operations():
    """Test ThoughtTree data structure operations."""
    tree = ThoughtTree()
    
    # Add root thought
    root_thought = Thought(
        type=ThoughtType.ANALYZE,
        content="Root thought",
        confidence=0.9,
        supporting_facts=["Initial fact"],
        next_action=None
    )
    root_idx = tree.add_node(root_thought, score=0.9)
    
    # Add child thoughts
    child1_thought = Thought(
        type=ThoughtType.ANALYZE,
        content="Child 1",
        confidence=0.8,
        supporting_facts=["Fact 1"],
        next_action=None
    )
    child1_idx = tree.add_node(child1_thought, parent=root_idx, score=0.8)
    
    child2_thought = Thought(
        type=ThoughtType.CONCLUDE,
        content="Child 2",
        confidence=0.7,
        supporting_facts=["Fact 2"],
        next_action=None
    )
    child2_idx = tree.add_node(child2_thought, parent=root_idx, score=0.7, is_terminal=True)
    
    # Test tree structure
    assert len(tree.nodes) == 3
    assert tree.nodes[root_idx].children == [child1_idx, child2_idx]
    assert tree.nodes[child1_idx].parent == root_idx
    assert tree.nodes[child2_idx].parent == root_idx
    
    # Test path finding
    path = tree.get_path_to_node(child2_idx)
    assert path == [root_idx, child2_idx]
    
    # Test best path
    best_path = tree.get_best_path()
    assert best_path == [root_idx, child1_idx]  # Child 1 has higher score 