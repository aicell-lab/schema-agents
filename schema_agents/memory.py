from __future__ import annotations

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
from sentence_transformers import SentenceTransformer
from schema_agents.schema_reasoning import MemoryEntry, MemoryType

class SemanticMemory(BaseModel):
    """Enhanced memory system with semantic search capabilities."""
    entries: List[MemoryEntry] = Field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    embedding_model: Optional[SentenceTransformer] = None
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = np.array([])

    def _get_entry_text(self, entry: MemoryEntry) -> str:
        """Convert memory entry to searchable text."""
        if isinstance(entry.content, str):
            return entry.content
        elif isinstance(entry.content, dict):
            return " ".join(str(v) for v in entry.content.values())
        elif isinstance(entry.content, list):
            return " ".join(str(x) for x in entry.content)
        return str(entry.content)

    def add_entry(self, entry_type: MemoryType, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new memory entry with semantic embedding."""
        entry = MemoryEntry(
            type=entry_type,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Get embedding for the entry
        entry_text = self._get_entry_text(entry)
        new_embedding = self.embedding_model.encode([entry_text])[0]
        
        # Add entry and its embedding
        self.entries.append(entry)
        if self.embeddings.size == 0:
            self.embeddings = new_embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embedding])

    def search(
        self, 
        query: str, 
        limit: int = 5,
        min_similarity: float = 0.3,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search memory entries by semantic similarity."""
        if not self.entries:
            return []

        # Get query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get indices of top matches
        top_indices = np.argsort(similarities)[::-1]
        
        # Filter and return results
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            entry = self.entries[idx]
            
            # Apply filters
            if similarity < min_similarity:
                continue
            if memory_types and entry.type not in memory_types:
                continue
                
            results.append((entry, float(similarity)))
            if len(results) >= limit:
                break
                
        return results

    def get_summary(
        self,
        query: Optional[str] = None,
        limit: int = 5,
        memory_types: Optional[List[MemoryType]] = None
    ) -> str:
        """Get a formatted summary of relevant memories."""
        if query:
            entries = [entry for entry, _ in self.search(query, limit, memory_types=memory_types)]
        else:
            entries = sorted(
                [e for e in self.entries if not memory_types or e.type in memory_types],
                key=lambda x: x.timestamp,
                reverse=True
            )[:limit]
            
        summary_parts = []
        for entry in entries:
            summary_parts.append(f"{entry.type.upper()}: {str(entry.content)[:200]}")
        return "\n".join(summary_parts)

    def get_facts(self) -> List[str]:
        """Get all established facts."""
        return [
            str(entry.content) 
            for entry in self.entries 
            if entry.type == MemoryType.FACT
        ]

    def get_type_summary(self, memory_type: MemoryType, limit: int = 5) -> str:
        """Get summary of specific memory type."""
        return self.get_summary(memory_types=[memory_type], limit=limit)

    def clear(self) -> None:
        """Clear all memories."""
        self.entries = []
        self.embeddings = np.array([]) 