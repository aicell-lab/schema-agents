"""Test utilities and shared fixtures."""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class TestDependencies(BaseModel):
    """Test dependencies for agent"""
    history: List[str] = Field(default_factory=list)
    context: Dict[str, str] = Field(default_factory=dict)
    vector_store: Optional[Dict] = Field(default_factory=dict)
    tool_calls: Dict[str, int] = Field(default_factory=dict)
    
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