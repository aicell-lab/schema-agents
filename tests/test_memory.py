import pytest
import numpy as np
from typing import List, Dict, Any
from schema_agents.memory import SemanticMemory
from schema_agents.schema_reasoning import MemoryEntry, MemoryType

@pytest.fixture
def memory():
    """Create a test memory instance."""
    return SemanticMemory()

@pytest.mark.asyncio
async def test_memory_basic_operations(memory):
    """Test basic memory operations."""
    # Add entries
    memory.add_entry(MemoryType.FACT, "The sky is blue")
    memory.add_entry(MemoryType.OBSERVATION, "It's currently sunny")
    memory.add_entry(MemoryType.THOUGHT, "Weather seems nice today")
    
    # Verify entries were added
    assert len(memory.entries) == 3
    assert memory.embeddings.shape == (3, 384)  # MiniLM embeddings are 384-dimensional
    
    # Test getting facts
    facts = memory.get_facts()
    assert len(facts) == 1
    assert "sky is blue" in facts[0]

@pytest.mark.asyncio
async def test_memory_semantic_search(memory):
    """Test semantic search functionality."""
    # Add test entries
    memory.add_entry(MemoryType.FACT, "Python is a programming language")
    memory.add_entry(MemoryType.FACT, "JavaScript runs in browsers")
    memory.add_entry(MemoryType.OBSERVATION, "The code has a bug")
    memory.add_entry(MemoryType.THOUGHT, "We should refactor this module")
    
    # Search for programming related entries
    results = memory.search("programming languages", limit=2)
    assert len(results) == 2
    assert any("Python" in entry.content for entry, _ in results)
    assert all(similarity >= 0.3 for _, similarity in results)
    
    # Search with type filter
    results = memory.search(
        "programming languages",
        memory_types=[MemoryType.FACT],
        limit=2
    )
    assert len(results) == 2
    assert all(entry.type == MemoryType.FACT for entry, _ in results)

@pytest.mark.asyncio
async def test_memory_structured_content(memory):
    """Test handling structured content in memory."""
    # Add structured content
    memory.add_entry(
        MemoryType.FACT,
        {"language": "Python", "version": "3.9", "purpose": "Backend development"}
    )
    memory.add_entry(
        MemoryType.OBSERVATION,
        ["test failed", "coverage: 80%", "3 errors found"]
    )
    
    # Search in structured content
    results = memory.search("Python programming")
    assert len(results) > 0
    assert any("Python" in str(entry.content) for entry, _ in results)
    
    # Get summary
    summary = memory.get_summary(limit=2)
    assert "Python" in summary
    assert "test failed" in summary

@pytest.mark.asyncio
async def test_memory_metadata(memory):
    """Test memory entry metadata handling."""
    # Add entries with metadata
    memory.add_entry(
        MemoryType.FACT,
        "Python is strongly typed",
        metadata={"source": "documentation", "confidence": 0.95}
    )
    memory.add_entry(
        MemoryType.OBSERVATION,
        "Test suite completed",
        metadata={"duration": "2.5s", "passed": True}
    )
    
    # Verify metadata is preserved
    assert memory.entries[0].metadata["source"] == "documentation"
    assert memory.entries[0].metadata["confidence"] == 0.95
    assert memory.entries[1].metadata["duration"] == "2.5s"
    assert memory.entries[1].metadata["passed"] is True

@pytest.mark.asyncio
async def test_memory_type_filtering(memory):
    """Test memory type filtering and summaries."""
    # Add mixed entries
    memory.add_entry(MemoryType.FACT, "Fact 1")
    memory.add_entry(MemoryType.FACT, "Fact 2")
    memory.add_entry(MemoryType.OBSERVATION, "Observation 1")
    memory.add_entry(MemoryType.THOUGHT, "Thought 1")
    memory.add_entry(MemoryType.THOUGHT, "Thought 2")
    
    # Test type-specific summaries
    fact_summary = memory.get_type_summary(MemoryType.FACT)
    assert "Fact 1" in fact_summary
    assert "Fact 2" in fact_summary
    assert "Observation" not in fact_summary
    
    thought_summary = memory.get_type_summary(MemoryType.THOUGHT)
    assert "Thought 1" in thought_summary
    assert "Thought 2" in thought_summary
    assert "Fact" not in thought_summary

@pytest.mark.asyncio
async def test_memory_clear(memory):
    """Test memory clearing."""
    # Add some entries
    memory.add_entry(MemoryType.FACT, "Test fact")
    memory.add_entry(MemoryType.OBSERVATION, "Test observation")
    
    # Verify entries exist
    assert len(memory.entries) == 2
    assert memory.embeddings.size > 0
    
    # Clear memory
    memory.clear()
    
    # Verify everything is cleared
    assert len(memory.entries) == 0
    assert memory.embeddings.size == 0 