import pytest
from typing import List, Dict, Any
from schema_agents.visualization import MemoryVisualizer, visualize_memory_state
from schema_agents.schema import (
    Action,
    ActionType,
    MemoryEntry,
    MemoryType,
    Observation,
    ReasoningState,
    Thought,
    ThoughtType
)
from schema_agents.schema_reasoning import (
    ReasoningState, Thought, Action, Observation,
    ThoughtType, ActionType, MemoryType
)

def create_test_state() -> ReasoningState:
    """Create a test reasoning state."""
    state = ReasoningState(task="Test task")
    
    # Add thoughts
    state.thoughts = [
        Thought(
            type=ThoughtType.ANALYZE,
            content="Initial analysis",
            confidence=0.9,
            supporting_facts=["Fact 1", "Fact 2"],
            next_action="calculate"
        ),
        Thought(
            type=ThoughtType.DECIDE,
            content="Make decision",
            confidence=0.8,
            supporting_facts=["Fact 3"],
            next_action="query"
        ),
        Thought(
            type=ThoughtType.CONCLUDE,
            content="Final conclusion",
            confidence=0.95,
            supporting_facts=["Fact 4", "Fact 5"],
            next_action=None
        )
    ]
    
    # Add actions
    state.actions = [
        Action(
            type=ActionType.TOOL_CALL,
            tool_name="calculate",
            arguments={"x": 1, "y": 2},
            purpose="Test calculation",
            expected_outcome="Sum",
            fallback=None
        ),
        Action(
            type=ActionType.QUERY,
            tool_name="search",
            arguments={"query": "test"},
            purpose="Find information",
            expected_outcome="Search results",
            fallback=None
        )
    ]
    
    # Add observations
    state.observations = [
        Observation(
            content="Result: 3",
            source="calculator",
            timestamp=1234567890
        ),
        Observation(
            content="Found relevant info",
            source="search",
            timestamp=1234567891
        )
    ]
    
    # Add memory entries
    state.memory = [
        MemoryEntry(
            type=MemoryType.FACT,
            content="Important fact",
            timestamp=1234567880,
            metadata={"source": "knowledge base"}
        ),
        MemoryEntry(
            type=MemoryType.OBSERVATION,
            content="Key observation",
            timestamp=1234567885,
            metadata={"confidence": 0.9}
        )
    ]
    
    return state

def test_memory_graph():
    """Test memory graph visualization."""
    state = create_test_state()
    graph = MemoryVisualizer.create_memory_graph(state)
    
    # Verify graph structure
    assert "graph TD" in graph
    assert "Task" in graph
    assert "thought_0" in graph
    assert "thought_1" in graph
    assert "thought_2" in graph
    assert "action_0" in graph
    assert "action_1" in graph
    assert "obs_0" in graph
    assert "obs_1" in graph
    
    # Verify connections
    assert "Task --> thought_0" in graph
    assert "thought_0 --> action_0" in graph
    assert "action_0 --> obs_0" in graph

def test_memory_timeline():
    """Test memory timeline visualization."""
    state = create_test_state()
    timeline = MemoryVisualizer.create_memory_timeline(state)
    
    # Verify timeline structure
    assert "gantt" in timeline
    assert "Memory Timeline" in timeline
    assert "Memory Events" in timeline
    assert "FACT" in timeline
    assert "OBSERVATION" in timeline

def test_state_diagram():
    """Test state diagram visualization."""
    state = create_test_state()
    diagram = MemoryVisualizer.create_state_diagram(state)
    
    # Verify diagram structure
    assert "stateDiagram-v2" in diagram
    assert "[*] --> Task" in diagram
    assert "ANALYZE" in diagram
    assert "DECIDE" in diagram
    assert "CONCLUDE" in diagram
    assert "Observation" in diagram

def test_confidence_chart():
    """Test confidence chart visualization."""
    state = create_test_state()
    chart = MemoryVisualizer.create_confidence_chart(state)
    
    # Verify chart structure
    assert "xychart-beta" in chart
    assert "Confidence Over Time" in chart
    assert "0.9" in chart  # First thought confidence
    assert "0.8" in chart  # Second thought confidence
    assert "0.95" in chart  # Third thought confidence

def test_visualize_memory_state():
    """Test combined visualization function."""
    state = create_test_state()
    
    # Test individual formats
    graph = visualize_memory_state(state, format="graph")
    assert "memory_graph" in graph
    assert "graph TD" in graph["memory_graph"]
    
    timeline = visualize_memory_state(state, format="timeline")
    assert "memory_timeline" in timeline
    assert "gantt" in timeline["memory_timeline"]
    
    state_diagram = visualize_memory_state(state, format="state")
    assert "state_diagram" in state_diagram
    assert "stateDiagram-v2" in state_diagram["state_diagram"]
    
    confidence = visualize_memory_state(state, format="confidence")
    assert "confidence_chart" in confidence
    assert "xychart-beta" in confidence["confidence_chart"]
    
    # Test all formats
    all_viz = visualize_memory_state(state, format="all")
    assert "memory_graph" in all_viz
    assert "memory_timeline" in all_viz
    assert "state_diagram" in all_viz
    assert "confidence_chart" in all_viz 