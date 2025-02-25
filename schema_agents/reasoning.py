from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, Dict, List, Literal, Optional, Union, AsyncIterator
from pydantic import BaseModel, Field
from pydantic_graph import BaseNode, Graph, GraphRunContext, GraphRun
from pydantic_graph.nodes import End

from pydantic_ai import RunContext, models, result, exceptions
from pydantic_ai.tools import Tool
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart, ToolCallPart, ModelResponse, ToolReturnPart, TextPart
from pydantic_ai._agent_graph import _prepare_request_parameters
from pydantic_ai.usage import Usage, UsageLimits
from pydantic_ai.settings import ModelSettings

# Base Reasoning Configs
class ReActConfig(BaseModel):
    """Configuration for ReAct reasoning."""
    max_loops: int = Field(5, description="Maximum reasoning cycles")
    min_confidence: float = Field(0.8, description="Minimum confidence threshold")
    timeout: int = Field(60, description="Maximum time in seconds")

class CoTConfig(BaseModel):
    """Configuration for Chain of Thought reasoning."""
    max_steps: int = Field(10, description="Maximum reasoning steps")
    step_prefix: str = Field("Step", description="Prefix for each step")

class ToTConfig(BaseModel):
    """Configuration for Tree of Thoughts reasoning."""
    max_branches: int = Field(3, description="Maximum parallel branches")
    max_depth: int = Field(3, description="Maximum depth per branch")
    min_branch_score: float = Field(0.4, description="Minimum score to explore")

class SRConfig(BaseModel):
    """Configuration for Self-Reflection reasoning."""
    reflection_rounds: int = Field(2, description="Number of improvement iterations")
    reflection_criteria: List[str] = Field(
        default_factory=lambda: ["clarity", "accuracy", "completeness"],
        description="Aspects to reflect on"
    )

class ReasoningStrategy(BaseModel):
    """Container for reasoning strategy configuration."""
    type: Union[Literal["react", "cot", "tot", "sr"], List[Literal["react", "cot", "tot", "sr"]]]
    react_config: Optional[ReActConfig] = None
    cot_config: Optional[CoTConfig] = None
    tot_config: Optional[ToTConfig] = None
    sr_config: Optional[SRConfig] = None
    verbose: bool = Field(True, description="Show reasoning steps")
    max_tokens: Optional[int] = Field(2000, description="Maximum response length")
    temperature: float = Field(0.7, description="Response creativity")

@dataclasses.dataclass
class ReasoningState:
    """State for reasoning graph execution."""
    message_history: list[ModelMessage]
    usage: Usage
    retries: int
    run_step: int
    loop_count: int = 0
    confidence: float = 0.0
    last_observation: Optional[str] = None
    final_answer: Optional[str] = None

    def increment_retries(self, max_result_retries: int) -> None:
        self.retries += 1
        if self.retries > max_result_retries:
            raise exceptions.UnexpectedModelBehavior(
                f'Exceeded maximum retries ({max_result_retries}) for result validation'
            )

@dataclasses.dataclass
class ReasoningDeps:
    """Dependencies for reasoning graph execution."""
    user_deps: Any
    prompt: str
    model: models.Model
    tools: List[Tool]
    strategy: ReasoningStrategy
    run_context: RunContext
    function_tools: Dict[str, Tool]
    model_settings: ModelSettings | None = None
    usage_limits: UsageLimits | None = None
    max_result_retries: int = 3
    end_strategy: Literal['early', 'exhaustive'] = 'early'
    result_schema: result.ResultSchema | None = None
    result_tools: List[Any] = dataclasses.field(default_factory=list)
    result_validators: List[Any] = dataclasses.field(default_factory=list)

# ReAct Graph Nodes
@dataclasses.dataclass
class ThoughtNode(BaseNode[ReasoningState, ReasoningDeps, str]):
    """Node for generating thoughts about the current state."""
    
    async def run(
        self, 
        ctx: GraphRunContext[ReasoningState, ReasoningDeps]
    ) -> Union[ActionNode, FinalAnswerNode]:
        # Track progress and intermediate results
        if not hasattr(ctx.state, 'intermediate_results'):
            ctx.state.intermediate_results = []
        
        # Add the last observation to intermediate results if it exists and isn't already included
        if ctx.state.last_observation and ctx.state.last_observation not in ctx.state.intermediate_results:
            ctx.state.intermediate_results.append(ctx.state.last_observation)
        
        # Generate thought about current state
        thought_prompt = (
            f"Current query: {ctx.deps.prompt}\n"
            f"Previous observation: {ctx.state.last_observation}\n"
            f"Progress so far: {', '.join(ctx.state.intermediate_results)}\n"
            "What should I do next? Think step by step and when you have the Final Response, start your response with 'Final Response:'. "
            "You can call multiple tools in parallel if needed."
        )
        
        request = ModelRequest(parts=[UserPromptPart(thought_prompt)])
        
        # Clean up message history to ensure tool calls are properly paired with responses
        cleaned_history = []
        pending_tool_calls = {}
        
        for msg in ctx.state.message_history:
            if isinstance(msg, ModelRequest):
                # Check for tool returns
                tool_returns = [p for p in msg.parts if isinstance(p, ToolReturnPart)]
                if tool_returns:
                    for tool_return in tool_returns:
                        if tool_return.tool_call_id in pending_tool_calls:
                            # Add both the tool call and its response
                            cleaned_history.append(pending_tool_calls[tool_return.tool_call_id])
                            cleaned_history.append(msg)
                            del pending_tool_calls[tool_return.tool_call_id]
                else:
                    # Check for tool calls
                    tool_calls = [p for p in msg.parts if isinstance(p, ToolCallPart)]
                    if tool_calls:
                        for tool_call in tool_calls:
                            pending_tool_calls[tool_call.tool_call_id] = msg
                    else:
                        # Regular message
                        cleaned_history.append(msg)
        
        # Update the message history
        ctx.state.message_history = cleaned_history
        ctx.state.message_history.append(request)
        
        # Check usage limits if any
        response, usage = await ctx.deps.model.request(
            ctx.state.message_history,
            model_settings={"temperature": ctx.deps.strategy.temperature},
            model_request_parameters=await _prepare_request_parameters(ctx)
        )
        
        # Update usage and message history
        ctx.state.usage.incr(usage)
        ctx.state.message_history.append(response)
        
        # Check if we have tool calls or Final Response
        if response.parts:
            # First check for tool calls
            tool_calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
            if tool_calls:
                # Return the first tool call - ActionNode will handle all pending calls
                return ActionNode(tool_calls[0])
            
            # Then check for text responses
            text_parts = [p for p in response.parts if isinstance(p, TextPart)]
            if text_parts:
                content = text_parts[0].content
                if content.startswith("Final Response:"):
                    return FinalAnswerNode(content)
                elif "Final Response:" in content:
                    # Extract Final Response from the content
                    final_answer = content.split("Final Response:")[1].strip()
                    return FinalAnswerNode(f"Final Response: {final_answer}")
                else:
                    # No Final Response yet, check for any remaining tool calls
                    if tool_calls:
                        return ActionNode(tool_calls[0])
                    # If no tool calls found, treat as Final Response
                    return FinalAnswerNode(content)
        
        # No valid response parts, treat as error
        return FinalAnswerNode("Error: No valid response from model")

@dataclasses.dataclass
class ActionNode(BaseNode[ReasoningState, ReasoningDeps, str]):
    """Node for executing actions based on thoughts."""
    tool_call: ToolCallPart
    
    async def run(
        self, 
        ctx: GraphRunContext[ReasoningState, ReasoningDeps]
    ) -> ObservationNode:
        # Get all pending tool calls from the message history
        tool_calls = []
        for msg in ctx.state.message_history:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        tool_calls.append(part)
        
        # Add the current tool call if not already included
        if self.tool_call not in tool_calls:
            tool_calls.append(self.tool_call)
        
        async def execute_tool(tool_call: ToolCallPart) -> tuple[str, ToolCallPart]:
            """Execute a single tool and return its result with the original call."""
            if tool := ctx.deps.function_tools.get(tool_call.tool_name):
                try:
                    # Create a run context for the tool
                    tool_ctx = RunContext(
                        deps=ctx.deps.user_deps,
                        model=ctx.deps.model,
                        usage=ctx.state.usage,
                        prompt=ctx.deps.prompt,
                        messages=ctx.state.message_history,
                        run_step=ctx.state.run_step
                    )
                    result = await tool.run(tool_call, tool_ctx)
                    return str(result.content), tool_call
                except Exception as e:
                    return f"Error executing {tool_call.tool_name}: {str(e)}", tool_call
            return f"Unknown action: {tool_call.tool_name}", tool_call
        
        # Execute all tools in parallel
        results = await asyncio.gather(*[execute_tool(tc) for tc in tool_calls])
        
        # Process results and add them to message history
        all_observations = []
        for result_content, tool_call in results:
            # Add tool response to message history
            tool_response = ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name=tool_call.tool_name,
                        content=result_content,
                        tool_call_id=tool_call.tool_call_id
                    )
                ]
            )
            ctx.state.message_history.append(tool_response)
            all_observations.append(result_content)
        
        # Combine all observations into a single observation
        combined_observation = " | ".join(all_observations)
        return ObservationNode(combined_observation)

@dataclasses.dataclass
class ObservationNode(BaseNode[ReasoningState, ReasoningDeps, str]):
    """Node for recording observations from actions."""
    observation: str
    
    async def run(
        self, 
        ctx: GraphRunContext[ReasoningState, ReasoningDeps]
    ) -> Union[ThoughtNode, End[str]]:
        ctx.state.last_observation = self.observation
        ctx.state.loop_count += 1
        
        # Check loop count
        if ctx.state.loop_count >= ctx.deps.strategy.react_config.max_loops:
            # Create a comprehensive Final Response that includes all intermediate results
            if hasattr(ctx.state, 'intermediate_results') and ctx.state.intermediate_results:
                final_message = (
                    f"Reached maximum number of steps ({ctx.deps.strategy.react_config.max_loops}). " +
                    f"Results so far: {' '.join(ctx.state.intermediate_results)}. " +
                    f"Latest observation: {self.observation}"
                )
            else:
                final_message = (
                    f"Reached maximum number of steps ({ctx.deps.strategy.react_config.max_loops}). " +
                    f"Current observation: {self.observation}. " +
                    "Based on the steps taken so far, this appears to be the answer."
                )
            raise ValueError(f"Exceeded maximum loop count ({ctx.deps.strategy.react_config.max_loops}): {final_message}")
            
        return ThoughtNode()

@dataclasses.dataclass
class FinalAnswerNode(BaseNode[ReasoningState, ReasoningDeps, str]):
    """Node for producing the Final Response."""
    thought: str
    
    async def run(
        self, 
        ctx: GraphRunContext[ReasoningState, ReasoningDeps]
    ) -> End[str]:
        # Extract Final Response from thought
        answer_lines = [line for line in self.thought.split("\n") if line.startswith("Final Response:")]
        if not answer_lines:
            # If no "Final Response:" prefix, use the entire thought as the answer
            final_answer = self.thought
        else:
            final_answer = answer_lines[0].replace("Final Response:", "").strip()
            
        ctx.state.final_answer = final_answer
        return End(final_answer)

def build_react_graph() -> Graph[ReasoningState, ReasoningDeps, str]:
    """Build the ReAct reasoning graph."""
    # Ensure ThoughtNode is first in the nodes tuple to be used as the default starting node
    nodes = (ThoughtNode, ActionNode, ObservationNode, FinalAnswerNode)
    return Graph[ReasoningState, ReasoningDeps, str](
        nodes=nodes,
        name="ReAct",
        state_type=ReasoningState,
        run_end_type=str
    )

async def execute_react_reasoning_sync(
    prompt: str,
    model: models.Model,
    tools: List[Tool],
    strategy: ReasoningStrategy,
    run_context: RunContext,
) -> str:
    """Execute ReAct reasoning strategy synchronously."""
    # Initialize state and dependencies
    state = ReasoningState(
        message_history=[],
        usage=Usage(),
        retries=0,
        run_step=0,
        loop_count=0,
        confidence=0.0,
        last_observation=None,
        final_answer=None
    )
    
    # Convert tools list to function_tools dict
    function_tools = {tool.name: tool for tool in tools}
    
    deps = ReasoningDeps(
        user_deps=run_context.deps,
        prompt=prompt,
        model=model,
        tools=tools,
        strategy=strategy,
        run_context=run_context,
        function_tools=function_tools,
        usage_limits=UsageLimits(),
        model_settings=None
    )
    
    # Build and run the graph
    graph = build_react_graph()
    ctx = GraphRunContext(state=state, deps=deps)
    graph_result = await graph.run(ThoughtNode(), state=state, deps=deps)
    return graph_result

async def execute_react_reasoning_stream(
    prompt: str,
    model: models.Model,
    tools: List[Tool],
    strategy: ReasoningStrategy,
    run_context: RunContext,
) -> AsyncIterator[str]:
    """Execute ReAct reasoning strategy with streaming."""
    # Initialize state and dependencies
    state = ReasoningState(
        message_history=[],
        usage=Usage(),
        retries=0,
        run_step=0,
        loop_count=0,
        confidence=0.0,
        last_observation=None,
        final_answer=None
    )

    # Convert tools list to function_tools dict
    function_tools = {tool.name: tool for tool in tools}

    deps = ReasoningDeps(
        user_deps=run_context.deps,
        prompt=prompt,
        model=model,
        tools=tools,
        strategy=strategy,
        run_context=run_context,
        function_tools=function_tools,
        usage_limits=UsageLimits(),
        model_settings=None
    )

    # For streaming, we need to handle the graph execution step by step
    graph = build_react_graph()
    graph_run = await graph.run(ThoughtNode(), state=state, deps=deps)
    async for node_result in graph_run:
        # For ThoughtNode, yield the last message from history
        if isinstance(node_result, ThoughtNode):
            if state.message_history:
                yield state.message_history[-1]
        # For FinalAnswerNode, yield the final answer
        elif isinstance(node_result, FinalAnswerNode):
            yield state.final_answer

async def execute_react_reasoning(
    prompt: str,
    model: models.Model,
    tools: List[Tool],
    strategy: ReasoningStrategy,
    run_context: RunContext,
    stream: bool = False
) -> Union[str, AsyncIterator[str]]:
    """Execute ReAct reasoning strategy."""
    # Initialize state
    state = ReasoningState(
        message_history=[],
        usage=run_context.usage,
        retries=0,
        run_step=run_context.run_step,
        loop_count=0,
        confidence=0.0
    )
    
    # Initialize dependencies
    function_tools = {tool.name: tool for tool in tools}
    deps = ReasoningDeps(
        user_deps=run_context.deps,
        prompt=prompt,
        model=model,
        tools=tools,
        strategy=strategy,
        run_context=run_context,
        function_tools=function_tools,
        usage_limits=model.usage_limits if hasattr(model, 'usage_limits') else None
    )
    
    # Build and run graph
    graph = build_react_graph()
    
    if stream:
        # Create an async generator for streaming results
        async def stream_generator():
            graph_run = GraphRun(
                graph=graph,
                start_node=ThoughtNode(),
                history=[],
                state=state,
                deps=deps,
                auto_instrument=True
            )
            async for node_result in graph_run:
                if isinstance(node_result, str):
                    yield node_result
                elif isinstance(node_result, ThoughtNode):
                    # For ThoughtNode, yield the last message from history
                    if state.message_history:
                        last_msg = state.message_history[-1]
                        if isinstance(last_msg, ModelResponse):
                            yield last_msg
                elif isinstance(node_result, FinalAnswerNode):
                    # For FinalAnswerNode, yield the final answer
                    yield node_result.thought
        
        # Return the generator directly
        return stream_generator()
    else:
        # Run synchronously and return final result
        graph_result = await graph.run(ThoughtNode(), state=state, deps=deps)
        return graph_result.output

@dataclasses.dataclass
class StreamThoughtNode(BaseNode[ReasoningState, ReasoningDeps, str]):
    """Node for generating streaming thoughts about the current state."""
    
    async def run(
        self, 
        ctx: GraphRunContext[ReasoningState, ReasoningDeps]
    ) -> Union[ActionNode, FinalAnswerNode]:
        # Track progress and intermediate results
        if not hasattr(ctx.state, 'intermediate_results'):
            ctx.state.intermediate_results = []
        
        # Add the last observation to intermediate results if it exists and isn't already included
        if ctx.state.last_observation and ctx.state.last_observation not in ctx.state.intermediate_results:
            ctx.state.intermediate_results.append(ctx.state.last_observation)
        
        # Generate thought about current state
        thought_prompt = (
            f"Current query: {ctx.deps.prompt}\n"
            f"Previous observation: {ctx.state.last_observation}\n"
            f"Progress so far: {', '.join(ctx.state.intermediate_results)}\n"
            "What should I do next? Think step by step and when you have the Final Response, start your response with 'Final Response:'. "
            "You can call multiple tools in parallel if needed."
        )
        
        request = ModelRequest(parts=[UserPromptPart(thought_prompt)])
        
        # Clean up message history to ensure tool calls are properly paired with responses
        cleaned_history = []
        pending_tool_calls = {}
        
        for msg in ctx.state.message_history:
            if isinstance(msg, ModelRequest):
                # Check for tool returns
                tool_returns = [p for p in msg.parts if isinstance(p, ToolReturnPart)]
                if tool_returns:
                    for tool_return in tool_returns:
                        if tool_return.tool_call_id in pending_tool_calls:
                            # Add both the tool call and its response
                            cleaned_history.append(pending_tool_calls[tool_return.tool_call_id])
                            cleaned_history.append(msg)
                            del pending_tool_calls[tool_return.tool_call_id]
                else:
                    # Check for tool calls
                    tool_calls = [p for p in msg.parts if isinstance(p, ToolCallPart)]
                    if tool_calls:
                        for tool_call in tool_calls:
                            pending_tool_calls[tool_call.tool_call_id] = msg
                    else:
                        # Regular message
                        cleaned_history.append(msg)
        
        # Update the message history
        ctx.state.message_history = cleaned_history
        ctx.state.message_history.append(request)
        
        # Check usage limits if any
        response, usage = await ctx.deps.model.request(
            ctx.state.message_history,
            model_settings={"temperature": ctx.deps.strategy.temperature},
            model_request_parameters=await _prepare_request_parameters(ctx)
        )
        
        # Update usage and message history
        ctx.state.usage.incr(usage)
        ctx.state.message_history.append(response)
        
        # Check if we have tool calls or Final Response
        if response.parts:
            # First check for tool calls
            tool_calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
            if tool_calls:
                # Return the first tool call - ActionNode will handle all pending calls
                return ActionNode(tool_calls[0])
            
            # Then check for text responses
            text_parts = [p for p in response.parts if isinstance(p, TextPart)]
            if text_parts:
                content = text_parts[0].content
                if content.startswith("Final Response:"):
                    return FinalAnswerNode(content)
                elif "Final Response:" in content:
                    # Extract Final Response from the content
                    final_answer = content.split("Final Response:")[1].strip()
                    return FinalAnswerNode(f"Final Response: {final_answer}")
                else:
                    # No Final Response yet, check for any remaining tool calls
                    if tool_calls:
                        return ActionNode(tool_calls[0])
                    # If no tool calls found, treat as Final Response
                    return FinalAnswerNode(content)
        
        # No valid response parts, treat as error
        return FinalAnswerNode("Error: No valid response from model")
