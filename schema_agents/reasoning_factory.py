from __future__ import annotations

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field

from schema_agents.schema_reasoning import SchemaReasoning
from schema_agents.tree_reasoning import TreeReasoning
from schema_agents.chain_reasoning import ChainReasoning
from pydantic_ai import models, RunContext

class ReasoningConfig(BaseModel):
    """Configuration for reasoning strategies."""
    strategy: Union[
        Literal["react"],
        Literal["tree"],
        Literal["chain"],
        List[Literal["react", "tree", "chain"]]
    ] = "react"
    
    # ReAct config
    max_steps: int = Field(default=10, description="Maximum reasoning steps")
    planning_interval: int = Field(default=3, description="Steps between plan updates")
    reflection_interval: int = Field(default=5, description="Steps between reflections")
    min_confidence: float = Field(default=0.7, description="Minimum confidence threshold")
    
    # Tree of Thoughts config
    max_depth: int = Field(default=5, description="Maximum tree depth")
    beam_width: int = Field(default=3, description="Beam search width")
    max_branches: int = Field(default=5, description="Maximum branches per node")
    
    # Chain of Thought config
    branching_factor: int = Field(default=2, description="Branches per decision point")
    max_chain_branches: int = Field(default=3, description="Maximum total branches")

def create_reasoning_strategy(
    strategy_type: Union[str, List[str]],
    model: models.Model,
    tools: List[Any],
    config: Optional[ReasoningConfig] = None
) -> Union[SchemaReasoning, TreeReasoning, ChainReasoning]:
    """Create a reasoning strategy instance."""
    config = config or ReasoningConfig()
    
    if isinstance(strategy_type, list):
        # TODO: Implement strategy composition
        strategy_type = strategy_type[0]
    
    if strategy_type == "react":
        return SchemaReasoning(
            model=model,
            tools=tools,
            max_steps=config.max_steps,
            planning_interval=config.planning_interval,
            reflection_interval=config.reflection_interval,
            min_confidence=config.min_confidence
        )
    elif strategy_type == "tree":
        return TreeReasoning(
            model=model,
            tools=tools,
            max_depth=config.max_depth,
            beam_width=config.beam_width,
            min_score=config.min_confidence,
            max_branches=config.max_branches
        )
    elif strategy_type == "chain":
        return ChainReasoning(
            model=model,
            tools=tools,
            max_steps=config.max_steps,
            min_confidence=config.min_confidence,
            branching_factor=config.branching_factor,
            max_branches=config.max_chain_branches
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

async def execute_reasoning(
    task: str,
    model: models.Model,
    tools: List[Any],
    run_context: RunContext,
    config: Optional[ReasoningConfig] = None,
    stream: bool = False
) -> Any:
    """Execute reasoning with configured strategy."""
    config = config or ReasoningConfig()
    
    if isinstance(config.strategy, list):
        # TODO: Implement strategy composition
        results = []
        for strategy in config.strategy:
            reasoner = create_reasoning_strategy(strategy, model, tools, config)
            result = await reasoner.execute(task, run_context, stream=stream)
            results.append(result)
        return results[-1]  # Return last result for now
    else:
        reasoner = create_reasoning_strategy(config.strategy, model, tools, config)
        return await reasoner.execute(task, run_context, stream=stream) 