from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple
from pydantic import BaseModel, Field, ConfigDict

from schema_agents.schema_reasoning import (
    ThoughtType, ActionType, MemoryType,
    Thought, Action, Observation, Plan, ReasoningState
)
from schema_agents.memory import SemanticMemory
from pydantic_ai import RunContext, models, result, exceptions

class ChainLink(BaseModel):
    """A link in the chain of thought."""
    thought: Thought
    action: Optional[Action] = None
    observation: Optional[Observation] = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    model_config = ConfigDict(extra="allow")

class ThoughtChain(BaseModel):
    """Chain of thoughts with branching support."""
    links: List[ChainLink] = Field(default_factory=list)
    branches: Dict[int, List[int]] = Field(default_factory=dict)  # link_idx -> branch_indices
    model_config = ConfigDict(extra="allow")
    
    def add_link(
        self,
        thought: Thought,
        action: Optional[Action] = None,
        observation: Optional[Observation] = None,
        score: float = 0.0,
        branch_from: Optional[int] = None
    ) -> int:
        """Add a link to the chain and return its index."""
        idx = len(self.links)
        link = ChainLink(
            thought=thought,
            action=action,
            observation=observation,
            score=score
        )
        self.links.append(link)
        
        if branch_from is not None:
            if branch_from not in self.branches:
                self.branches[branch_from] = []
            self.branches[branch_from].append(idx)
            
        return idx
    
    def get_branch_points(self) -> List[int]:
        """Get indices of links that have branches."""
        return list(self.branches.keys())
    
    def get_best_branch(self, from_idx: int) -> Optional[int]:
        """Get highest scoring branch from a given link."""
        if from_idx not in self.branches:
            return None
            
        branches = self.branches[from_idx]
        if not branches:
            return None
            
        return max(branches, key=lambda x: self.links[x].score)

class ChainReasoning:
    """Chain of Thought reasoning implementation."""
    def __init__(
        self,
        model: models.Model,
        tools: List[Any],
        max_steps: int = 10,
        min_confidence: float = 0.7,
        branching_factor: int = 2,
        max_branches: int = 3
    ):
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.min_confidence = min_confidence
        self.branching_factor = branching_factor
        self.max_branches = max_branches
        self.memory = SemanticMemory()

    async def _evaluate_step(
        self,
        link: ChainLink,
        state: ReasoningState,
        run_context: RunContext
    ) -> float:
        """Evaluate a chain link's quality."""
        context = self.memory.get_summary(link.thought.content, limit=3)
        
        prompt = f"""Task: {state.task}

Context:
{context}

Step to evaluate:
{link.model_dump_json(indent=2)}

Evaluate this reasoning step's quality. Consider:
1. Logical progression
2. Use of evidence
3. Action effectiveness
4. Progress toward goal
5. Clarity of thought

Score between 0 and 1:"""

        response = await self.model.request(
            messages=[{"role": "user", "content": prompt}],
            result_type=float
        )
        return response.data

    async def _should_branch(
        self,
        link: ChainLink,
        state: ReasoningState,
        run_context: RunContext
    ) -> bool:
        """Determine if we should create alternative branches."""
        if link.score < 0.8:  # Only branch on promising but not perfect steps
            prompt = f"""Task: {state.task}

Step:
{link.model_dump_json(indent=2)}

Should we explore alternative approaches from this point? Consider:
1. Are there other valid perspectives?
2. Could different approaches work better?
3. Is this step uncertain?
4. Are there unexplored options?

Response (true/false):"""

            response = await self.model.request(
                messages=[{"role": "user", "content": prompt}],
                result_type=bool
            )
            return response.data
        return False

    async def _generate_next_thought(
        self,
        state: ReasoningState,
        prev_link: Optional[ChainLink],
        run_context: RunContext,
        is_branch: bool = False
    ) -> Thought:
        """Generate next thought in the chain."""
        context = self.memory.get_summary(state.task, limit=3)
        
        prompt = f"""Task: {state.task}

Context:
{context}

Previous step:
{prev_link.model_dump_json(indent=2) if prev_link else "None"}

Generate the next thought in the chain. {
    "Consider alternative approaches." if is_branch else ""
}

Response must be valid JSON matching:
{{
    "type": "one of: plan, analyze, decide, reflect, conclude",
    "content": "detailed thought process",
    "confidence": "float between 0 and 1",
    "supporting_facts": ["relevant facts"],
    "next_action": "optional next action"
}}"""

        response = await self.model.request(
            messages=[{"role": "user", "content": prompt}],
            result_type=Thought
        )
        return response.data

    async def _generate_action(
        self,
        thought: Thought,
        state: ReasoningState,
        run_context: RunContext
    ) -> Optional[Action]:
        """Generate action for a thought if needed."""
        if not thought.next_action:
            return None
            
        tool_descriptions = []
        for name, tool in self.tools.items():
            tool_def = await tool.prepare_tool_def(run_context)
            if tool_def:
                tool_descriptions.append(f"- {name}: {tool_def.description}")

        prompt = f"""Task: {state.task}

Available Tools:
{chr(10).join(tool_descriptions)}

Thought:
{thought.model_dump_json(indent=2)}

Generate action. Response must be valid JSON matching:
{{
    "type": "one of: tool_call, query, compute, delegate, respond",
    "tool_name": "name of tool to call (if applicable)",
    "arguments": "dictionary of tool arguments (if applicable)",
    "purpose": "why this action is needed",
    "expected_outcome": "what we expect from this action",
    "fallback": "optional fallback plan"
}}"""

        response = await self.model.request(
            messages=[{"role": "user", "content": prompt}],
            result_type=Action
        )
        return response.data

    async def execute(
        self,
        task: str,
        run_context: RunContext,
        stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """Execute chain of thought reasoning."""
        state = ReasoningState(task=task)
        chain = ThoughtChain()
        
        async def reasoning_generator() -> AsyncIterator[str]:
            # Generate initial thought
            thought = await self._generate_next_thought(state, None, run_context)
            action = await self._generate_action(thought, state, run_context)
            
            # Add initial link
            current_idx = chain.add_link(thought=thought, action=action)
            if stream:
                yield f"Initial thought: {thought.content[:100]}..."
            
            while len(chain.links) < self.max_steps:
                current_link = chain.links[current_idx]
                
                # Execute action if present
                observation = None
                if current_link.action:
                    try:
                        result = await self._execute_action(current_link.action, run_context)
                        observation = Observation(
                            content=str(result),
                            source=current_link.action.tool_name or "system",
                            timestamp=time.time()
                        )
                        if stream:
                            yield f"Action result: {str(result)[:100]}..."
                    except Exception as e:
                        observation = Observation(
                            content=f"Error: {str(e)}",
                            source="error",
                            timestamp=time.time()
                        )
                
                # Update link with observation
                current_link.observation = observation
                
                # Evaluate current step
                score = await self._evaluate_step(current_link, state, run_context)
                current_link.score = score
                
                if stream:
                    yield f"Step score: {score:.2f}"
                
                # Check if we should branch
                if (
                    await self._should_branch(current_link, state, run_context) and
                    len(chain.branches) < self.max_branches
                ):
                    if stream:
                        yield "Exploring alternative branches..."
                        
                    # Generate alternative thoughts
                    for _ in range(self.branching_factor):
                        alt_thought = await self._generate_next_thought(
                            state, current_link, run_context, is_branch=True
                        )
                        alt_action = await self._generate_action(alt_thought, state, run_context)
                        branch_idx = chain.add_link(
                            thought=alt_thought,
                            action=alt_action,
                            branch_from=current_idx
                        )
                        if stream:
                            yield f"Generated alternative: {alt_thought.content[:100]}..."
                
                # Check if thought is conclusive
                if current_link.thought.type == ThoughtType.CONCLUDE:
                    break
                
                # Generate next thought
                thought = await self._generate_next_thought(state, current_link, run_context)
                action = await self._generate_action(thought, state, run_context)
                current_idx = chain.add_link(thought=thought, action=action)
                
                if stream:
                    yield f"Next thought: {thought.content[:100]}..."
            
            # Construct final answer from best path
            answer_parts = []
            idx = 0
            while idx < len(chain.links):
                link = chain.links[idx]
                answer_parts.append(f"Step {len(answer_parts) + 1}: {link.thought.content}")
                
                # Check for better branch
                best_branch = chain.get_best_branch(idx)
                if best_branch is not None and chain.links[best_branch].score > link.score:
                    idx = best_branch
                else:
                    idx += 1
            
            final_answer = "\n".join(answer_parts)
            if stream:
                yield f"Final answer:\n{final_answer}"
            yield final_answer

        if stream:
            return reasoning_generator()
        else:
            final_answer = None
            async for response in reasoning_generator():
                final_answer = response
            return final_answer

    async def _execute_action(self, action: Action, run_context: RunContext) -> Any:
        """Execute an action and return the result."""
        try:
            if action.type == ActionType.TOOL_CALL and action.tool_name:
                tool = self.tools.get(action.tool_name)
                if not tool:
                    return f"Error: Tool {action.tool_name} not found"
                return await tool.run(run_context, **(action.arguments or {}))
            elif action.type == ActionType.RESPOND:
                return action.arguments.get("response", "") if action.arguments else ""
            else:
                return f"Unsupported action type: {action.type}"
        except Exception as e:
            return f"Error executing action: {str(e)}" 