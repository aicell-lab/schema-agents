#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, List, Dict, Union
import time

from pydantic import BaseModel, Field, ConfigDict


@dataclass
class Message:
    """list[<role>: <content>]"""
    content: str
    data: BaseModel = field(default=None)
    role: str = field(default='user')  # system / user / assistant
    cause_by: Callable = field(default=None)
    processed_by: set['Role'] = field(default_factory=set)
    session_id: str = field(default=None)
    session_history: list[str] = field(default_factory=list)

    def __str__(self):
        return f"{self.role}: {self.content}"

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content
        }

class RoleSetting(BaseModel):
    """Role setting"""
    name: str
    profile: str
    goal: str
    instructions: str
    constraints: Optional[str] = ""
    icon: Optional[str] = None

    def __str__(self):
        return f"{self.name}({self.profile})"

    def __repr__(self):
        return self.__str__()


class Session(BaseModel):
    id: Optional[str] = None
    event_bus: Optional[Any] = None
    role_setting: Optional[RoleSetting] = None
    stop: bool = False
    
    def model_dump(self, **kwargs):
        # Exclude event_bus from serialization since it's not JSON serializable
        kwargs.setdefault('exclude', set()).add('event_bus')
        return super().model_dump(**kwargs)


class StreamEvent(BaseModel):
    type: str
    query_id: str
    session: Optional[Session] = None
    status: str
    content: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None

    class Config:
        # This ensures proper JSON serialization
        json_encoders = {
            Session: lambda v: v.model_dump(mode="json") if v else None
        }

class ThoughtType(str, enum.Enum):
    """Types of thoughts in the reasoning process."""
    PLAN = "plan"  # Planning and strategy
    ANALYZE = "analyze"  # Analyzing information
    DECIDE = "decide"  # Making decisions
    REFLECT = "reflect"  # Self-reflection
    CONCLUDE = "conclude"  # Drawing conclusions

class ActionType(str, enum.Enum):
    """Types of actions the agent can take."""
    TOOL_CALL = "tool_call"  # Call an external tool
    QUERY = "query"  # Request information
    COMPUTE = "compute"  # Perform computation
    DELEGATE = "delegate"  # Delegate to another agent
    RESPOND = "respond"  # Generate response

class MemoryType(str, enum.Enum):
    """Types of memory entries."""
    FACT = "fact"  # Established facts
    PLAN = "plan"  # Planning steps
    OBSERVATION = "observation"  # Tool observations
    THOUGHT = "thought"  # Reasoning steps
    ACTION = "action"  # Actions taken
    ERROR = "error"  # Error records
    RESULT = "result"  # Final results

class Thought(BaseModel):
    """Schema for structured thoughts."""
    type: ThoughtType
    content: str = Field(..., description="The main thought content")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    supporting_facts: List[str] = Field(default_factory=list)
    next_action: Optional[str] = None
    model_config = ConfigDict(extra="allow")

class Action(BaseModel):
    """Schema for structured actions."""
    type: ActionType
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    purpose: str = Field(..., description="Why this action is needed")
    expected_outcome: str = Field(..., description="What we expect from this action")
    fallback: Optional[str] = None
    model_config = ConfigDict(extra="allow")

class Observation(BaseModel):
    """Schema for structured observations."""
    content: str = Field(..., description="The observation content")
    source: str = Field(..., description="Source of the observation")
    timestamp: float = Field(..., description="When the observation was made")
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    model_config = ConfigDict(extra="allow")

class Plan(BaseModel):
    """Schema for structured plans."""
    steps: List[str] = Field(..., description="Planned steps")
    current_step: int = Field(default=0, description="Current step index")
    estimated_steps: int = Field(..., description="Estimated total steps needed")
    success_criteria: List[str] = Field(..., description="Criteria for success")
    model_config = ConfigDict(extra="allow")

class MemoryEntry(BaseModel):
    """Schema for memory entries."""
    type: MemoryType
    content: Union[str, Dict[str, Any], List[Any]]
    timestamp: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")

class ReasoningState(BaseModel):
    """Schema for reasoning state."""
    task: str = Field(..., description="The original task")
    current_plan: Optional[Plan] = None
    thoughts: List[Thought] = Field(default_factory=list)
    actions: List[Action] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    memory: List[MemoryEntry] = Field(default_factory=list)
    step_count: int = Field(default=0)
    final_answer: Optional[str] = None
    model_config = ConfigDict(extra="allow")

    def add_memory(self, entry_type: MemoryType, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a memory entry."""
        self.memory.append(MemoryEntry(
            type=entry_type,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {}
        ))

    def get_relevant_memories(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Get relevant memories for a query."""
        # TODO: Implement semantic search
        return sorted(self.memory, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_memory_summary(self) -> str:
        """Get a summary of the current memory state."""
        summary_parts = []
        for entry in sorted(self.memory, key=lambda x: x.timestamp, reverse=True)[:5]:
            summary_parts.append(f"{entry.type.upper()}: {str(entry.content)[:200]}")
        return "\n".join(summary_parts)

class Session(BaseModel):
    """Schema for session state."""
    id: str
    role_setting: Optional[Dict[str, Any]] = None
    event_bus: Optional[Any] = None
    model_config = ConfigDict(extra="allow")
