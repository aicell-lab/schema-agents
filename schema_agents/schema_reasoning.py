from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union, AsyncIterator, Set, Type, Callable
from pydantic import BaseModel, Field, ConfigDict

from pydantic_ai import RunContext, models, result, exceptions
from pydantic_ai.tools import Tool
from pydantic_ai.messages import ModelMessage, TextPart, ToolCallPart

# Configure logging
logger = logging.getLogger("schema_reasoning")
logger.setLevel(logging.INFO)

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
        """Add an entry to memory."""
        import time
        self.memory.append(
            MemoryEntry(
                type=entry_type,
                content=content,
                timestamp=time.time(),
                metadata=metadata or {}
            )
        )

    def get_relevant_memories(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Get memories relevant to a query using semantic similarity."""
        # TODO: Implement semantic search
        return sorted(self.memory, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_memory_summary(self) -> str:
        """Get a formatted summary of relevant memories."""
        summary_parts = []
        for entry in self.memory[-5:]:  # Last 5 memories
            summary_parts.append(f"{entry.type.upper()}: {str(entry.content)[:200]}")
        return "\n".join(summary_parts)

class SchemaReasoning:
    """Enhanced schema-based reasoning implementation."""
    def __init__(
        self,
        model: models.Model,
        tools: List[Tool],
        max_steps: int = 10,
        planning_interval: int = 3,
        reflection_interval: int = 5,
        min_confidence: float = 0.7
    ):
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.planning_interval = planning_interval
        self.reflection_interval = reflection_interval
        self.min_confidence = min_confidence

    async def _generate_initial_plan(self, state: ReasoningState, run_context: RunContext) -> Plan:
        """Generate initial plan for the task."""
        prompt = f"""Task: {state.task}

Please analyze this task and create a structured plan. Your response must be valid JSON matching this schema:

{{
    "steps": ["list of specific steps"],
    "estimated_steps": "integer estimate of total steps needed",
    "success_criteria": ["list of specific success criteria"]
}}

Think carefully and be specific."""

        response = await self.model.request(
            messages=[{"role": "user", "content": prompt}],
            result_type=Plan
        )
        return response.data

    async def _generate_thought(self, state: ReasoningState, run_context: RunContext) -> Thought:
        """Generate next thought based on current state."""
        # Get relevant memory summary
        memory_summary = state.get_memory_summary()
        
        prompt = f"""Task: {state.task}

Current State:
{memory_summary}

Current Plan Step: {state.current_plan.current_step + 1 if state.current_plan else 0} of {state.current_plan.estimated_steps if state.current_plan else 'unknown'}

Generate the next reasoning step. Your response must be valid JSON matching this schema:

{{
    "type": "one of: plan, analyze, decide, reflect, conclude",
    "content": "your detailed thought process",
    "confidence": "float between 0 and 1",
    "supporting_facts": ["relevant facts that support this thought"],
    "next_action": "optional next action to take"
}}

Think carefully and be specific."""

        response = await self.model.request(
            messages=[{"role": "user", "content": prompt}],
            result_type=Thought
        )
        return response.data

    async def _generate_action(self, state: ReasoningState, thought: Thought, run_context: RunContext) -> Action:
        """Generate next action based on thought."""
        # Prepare tool descriptions
        tool_descriptions = []
        for name, tool in self.tools.items():
            tool_def = await tool.prepare_tool_def(run_context)
            if tool_def:
                tool_descriptions.append(f"- {name}: {tool_def.description}")

        prompt = f"""Task: {state.task}

Available Tools:
{chr(10).join(tool_descriptions)}

Current Thought:
{thought.model_dump_json(indent=2)}

Generate the next action to take. Your response must be valid JSON matching this schema:

{{
    "type": "one of: tool_call, query, compute, delegate, respond",
    "tool_name": "name of tool to call (if applicable)",
    "arguments": "dictionary of tool arguments (if applicable)",
    "purpose": "why this action is needed",
    "expected_outcome": "what we expect from this action",
    "fallback": "optional fallback plan"
}}

Think carefully and be specific."""

        response = await self.model.request(
            messages=[{"role": "user", "content": prompt}],
            result_type=Action
        )
        return response.data

    async def _execute_action(self, action: Action, run_context: RunContext) -> str:
        """Execute an action and return the observation."""
        try:
            if action.type == ActionType.TOOL_CALL and action.tool_name:
                tool = self.tools.get(action.tool_name)
                if not tool:
                    return f"Error: Tool {action.tool_name} not found"
                result = await tool.run(run_context, **(action.arguments or {}))
                return str(result)
            elif action.type == ActionType.RESPOND:
                return action.arguments.get("response", "") if action.arguments else ""
            else:
                return f"Unsupported action type: {action.type}"
        except Exception as e:
            return f"Error executing action: {str(e)}"

    async def _should_continue(self, state: ReasoningState, run_context: RunContext) -> bool:
        """Determine if reasoning should continue."""
        # Check max steps
        if state.step_count >= self.max_steps:
            return False

        # Check if we have a final answer with high confidence
        if state.final_answer:
            last_thought = state.thoughts[-1] if state.thoughts else None
            if last_thought and last_thought.confidence >= self.min_confidence:
                return False

        return True

    async def execute(
        self,
        task: str,
        run_context: RunContext,
        stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """Execute reasoning process."""
        state = ReasoningState(task=task)

        async def reasoning_generator() -> AsyncIterator[str]:
            # Generate initial plan
            plan = await self._generate_initial_plan(state, run_context)
            state.current_plan = plan
            state.add_memory(MemoryType.PLAN, plan)
            if stream:
                yield f"Generated initial plan with {len(plan.steps)} steps"

            while await self._should_continue(state, run_context):
                # Check if we need to update plan
                if state.step_count > 0 and state.step_count % self.planning_interval == 0:
                    thought = await self._generate_thought(state, run_context)
                    if thought.type == ThoughtType.PLAN:
                        state.current_plan = await self._generate_initial_plan(state, run_context)
                        state.add_memory(MemoryType.PLAN, state.current_plan)
                        if stream:
                            yield "Updated plan based on progress"

                # Generate thought
                thought = await self._generate_thought(state, run_context)
                state.thoughts.append(thought)
                state.add_memory(MemoryType.THOUGHT, thought)
                if stream:
                    yield f"Thought: {thought.content}"

                # Check if thought confidence is too low
                if thought.confidence < self.min_confidence:
                    if stream:
                        yield f"Low confidence ({thought.confidence}), generating another thought"
                    continue

                # Generate and execute action if needed
                if thought.next_action:
                    action = await self._generate_action(state, thought, run_context)
                    state.actions.append(action)
                    state.add_memory(MemoryType.ACTION, action)
                    if stream:
                        yield f"Action: {action.type} - {action.purpose}"

                    # Execute action
                    observation = await self._execute_action(action, run_context)
                    state.add_memory(MemoryType.OBSERVATION, observation)
                    if stream:
                        yield f"Observation: {observation}"

                # Check for final answer
                if thought.type == ThoughtType.CONCLUDE:
                    state.final_answer = thought.content
                    break

                state.step_count += 1

            # Return final answer
            final_response = state.final_answer or "Failed to reach conclusion"
            if stream:
                yield f"Final response: {final_response}"
            yield final_response

        if stream:
            return reasoning_generator()
        else:
            final_response = None
            async for response in reasoning_generator():
                final_response = response
            return final_response

    def visualize_memory(self, state: ReasoningState) -> str:
        """Generate a visualization of the memory state."""
        # TODO: Implement memory visualization
        return "Memory visualization not implemented" 