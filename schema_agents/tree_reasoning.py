from __future__ import annotations

import asyncio
import heapq
from typing import List, Dict, Any, Optional, Set, Tuple, AsyncIterator
from pydantic import BaseModel, Field, ConfigDict

from schema_agents.schema_reasoning import (
    ThoughtType, ActionType, MemoryType,
    Thought, Action, Observation, Plan, ReasoningState
)
from schema_agents.memory import SemanticMemory
from pydantic_ai import RunContext, models, result, exceptions

class ThoughtNode(BaseModel):
    """Node in the thought tree."""
    thought: Thought
    parent: Optional[int] = None  # Index of parent node
    children: List[int] = Field(default_factory=list)  # Indices of child nodes
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    depth: int = Field(default=0)
    is_terminal: bool = Field(default=False)
    model_config = ConfigDict(extra="allow")

class ThoughtTree(BaseModel):
    """Tree structure for tracking thoughts."""
    nodes: List[ThoughtNode] = Field(default_factory=list)
    root: int = Field(default=0)  # Index of root node
    current: int = Field(default=0)  # Index of current node
    model_config = ConfigDict(extra="allow")
    
    def add_node(
        self,
        thought: Thought,
        parent: Optional[int] = None,
        score: float = 0.0,
        is_terminal: bool = False
    ) -> int:
        """Add a node to the tree and return its index."""
        idx = len(self.nodes)
        node = ThoughtNode(
            thought=thought,
            parent=parent,
            score=score,
            depth=self.nodes[parent].depth + 1 if parent is not None else 0,
            is_terminal=is_terminal
        )
        self.nodes.append(node)
        
        if parent is not None:
            self.nodes[parent].children.append(idx)
            
        return idx
    
    def get_path_to_node(self, node_idx: int) -> List[int]:
        """Get path from root to given node."""
        path = []
        current = node_idx
        while current is not None:
            path.append(current)
            current = self.nodes[current].parent
        return list(reversed(path))
    
    def get_best_path(self) -> List[int]:
        """Get path to highest scoring terminal node."""
        best_score = -1
        best_node = None
        
        for i, node in enumerate(self.nodes):
            if node.is_terminal and node.score > best_score:
                best_score = node.score
                best_node = i
                
        return self.get_path_to_node(best_node) if best_node is not None else []

class TreeReasoning:
    """Tree of Thoughts reasoning implementation."""
    def __init__(
        self,
        model: models.Model,
        tools: List[Any],
        max_depth: int = 5,
        beam_width: int = 3,
        min_score: float = 0.5,
        max_branches: int = 5
    ):
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.min_score = min_score
        self.max_branches = max_branches
        self.memory = SemanticMemory()

    async def _evaluate_thought(
        self,
        thought: Thought,
        state: ReasoningState,
        run_context: RunContext
    ) -> float:
        """Evaluate a thought's quality."""
        # Get relevant context
        context = self.memory.get_summary(thought.content, limit=3)
        
        prompt = f"""Task: {state.task}

Context:
{context}

Thought to evaluate:
{thought.model_dump_json(indent=2)}

Evaluate this thought's quality and relevance. Your response must be a float between 0 and 1, where:
- 1.0: Perfect, directly leads to solution
- 0.8-0.9: Excellent progress
- 0.6-0.7: Good progress
- 0.4-0.5: Moderate progress
- 0.2-0.3: Limited progress
- 0.0-0.1: No progress or irrelevant

Consider:
1. Relevance to task
2. Logical reasoning
3. Use of context
4. Actionability
5. Potential impact

Response (just the number):"""

        response = await self.model.request(
            messages=[{"role": "user", "content": prompt}],
            result_type=float
        )
        return response.data

    async def _generate_thoughts(
        self,
        state: ReasoningState,
        parent_thought: Optional[Thought],
        run_context: RunContext,
        num_thoughts: int = 3
    ) -> List[Thought]:
        """Generate multiple alternative thoughts."""
        # Get relevant context
        context = self.memory.get_summary(state.task, limit=3)
        
        prompt = f"""Task: {state.task}

Context:
{context}

Previous thought:
{parent_thought.model_dump_json(indent=2) if parent_thought else "None"}

Generate {num_thoughts} different thoughts for the next step. Each thought must be valid JSON matching this schema:

{{
    "type": "one of: plan, analyze, decide, reflect, conclude",
    "content": "detailed thought process",
    "confidence": "float between 0 and 1",
    "supporting_facts": ["relevant facts"],
    "next_action": "optional next action"
}}

Ensure thoughts are diverse and explore different approaches.

Thoughts (as JSON array):"""

        response = await self.model.request(
            messages=[{"role": "user", "content": prompt}],
            result_type=List[Thought]
        )
        return response.data

    async def _is_terminal(
        self,
        thought: Thought,
        state: ReasoningState,
        run_context: RunContext
    ) -> bool:
        """Determine if a thought should be terminal."""
        if thought.type == ThoughtType.CONCLUDE:
            return True
            
        prompt = f"""Task: {state.task}

Thought:
{thought.model_dump_json(indent=2)}

Should this be a terminal (final) thought? Consider:
1. Does it fully address the task?
2. Is further exploration needed?
3. Is it a conclusive response?

Response (true/false):"""

        response = await self.model.request(
            messages=[{"role": "user", "content": prompt}],
            result_type=bool
        )
        return response.data

    async def execute(
        self,
        task: str,
        run_context: RunContext,
        stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """Execute tree of thoughts reasoning."""
        state = ReasoningState(task=task)
        tree = ThoughtTree()
        
        async def reasoning_generator() -> AsyncIterator[str]:
            # Generate initial thoughts
            thoughts = await self._generate_thoughts(state, None, run_context, self.max_branches)
            
            # Add initial thoughts to tree
            for thought in thoughts:
                score = await self._evaluate_thought(thought, state, run_context)
                is_terminal = await self._is_terminal(thought, state, run_context)
                tree.add_node(thought, score=score, is_terminal=is_terminal)
                
                if stream:
                    yield f"Generated initial thought: {thought.content[:100]}... (score: {score:.2f})"
            
            # Beam search through thought tree
            depth = 0
            while depth < self.max_depth:
                # Get best nodes at current depth
                nodes_at_depth = [
                    (i, node) for i, node in enumerate(tree.nodes)
                    if node.depth == depth and node.score >= self.min_score and not node.is_terminal
                ]
                
                if not nodes_at_depth:
                    break
                    
                # Sort by score and take top k
                nodes_at_depth.sort(key=lambda x: x[1].score, reverse=True)
                beam = nodes_at_depth[:self.beam_width]
                
                # Expand each node in beam
                for node_idx, node in beam:
                    thoughts = await self._generate_thoughts(
                        state, node.thought, run_context, self.max_branches
                    )
                    
                    for thought in thoughts:
                        score = await self._evaluate_thought(thought, state, run_context)
                        is_terminal = await self._is_terminal(thought, state, run_context)
                        child_idx = tree.add_node(
                            thought,
                            parent=node_idx,
                            score=score,
                            is_terminal=is_terminal
                        )
                        
                        if stream:
                            yield f"Generated thought at depth {depth + 1}: {thought.content[:100]}... (score: {score:.2f})"
                
                depth += 1
            
            # Get best path and construct final answer
            best_path = tree.get_best_path()
            if not best_path:
                yield "Failed to reach conclusion"
                return
                
            # Construct answer from path
            answer_parts = []
            for idx in best_path:
                node = tree.nodes[idx]
                answer_parts.append(f"Step {node.depth + 1}: {node.thought.content}")
            
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