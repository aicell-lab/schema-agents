from __future__ import annotations

import asyncio
import traceback
import dataclasses
import enum
import inspect
import json
import logging
import re
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union, AsyncIterator, Set, Type, Callable, TypeVar, cast
from pydantic import BaseModel, Field
from pydantic_graph import BaseNode, Graph, GraphRunContext, GraphRun
from pydantic_graph.nodes import End

from pydantic_ai import RunContext, models, result, exceptions
from pydantic_ai.tools import Tool
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart, ToolCallPart, ModelResponse, ToolReturnPart, TextPart
from pydantic_ai._agent_graph import _prepare_request_parameters
from pydantic_ai.usage import Usage, UsageLimits
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import ModelRequestParameters
from openai import AsyncOpenAI

# Configure logging
logger = logging.getLogger("execute_simple_react")
logger.setLevel(logging.INFO)

# Define GraphResult type explicitly to avoid Union instantiation errors
GraphResult = Any  # This avoids Union instantiation issues

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
    intermediate_results: List[Any] = dataclasses.field(default_factory=list)
    memory: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    def increment_retries(self, max_result_retries: int) -> None:
        self.retries += 1
        if self.retries > max_result_retries:
            raise exceptions.UnexpectedModelBehavior(
                f'Exceeded maximum retries ({max_result_retries}) for result validation'
            )
    
    def add_memory_entry(self, entry_type: str, content: Any) -> None:
        """Add an entry to the reasoning memory."""
        self.memory.append({
            "type": entry_type,
            "content": content,
            "step": self.loop_count
        })
    
    def get_memory_summary(self) -> str:
        """Get a summary of the memory for prompt enhancement."""
        if not self.memory:
            return ""
        
        summary = "Previous steps:\n"
        for entry in self.memory:
            summary += f"Step {entry['step']}: {entry['type']} - {str(entry['content'])[:100]}...\n"
        return summary

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
class ThoughtNode(BaseNode):
    """Generate reasoning and output an action or final answer."""
    final_answer: bool = False
    
    async def run(
        self,
        ctx: GraphRunContext[ReasoningState, ReasoningDeps]
    ) -> GraphResult:  # Use GraphResult instead of Union
        """Generate reasoning and decide what to do next."""
        # If this node is marked as final_answer, return a FinalAnswerNode with the state's final answer
        if self.final_answer and ctx.state.final_answer:
            return FinalAnswerNode(f"Final Response: {ctx.state.final_answer}")
            
        # Track progress and intermediate results
        if ctx.state.loop_count > 0:
            # Add memory about the previous observation
            if ctx.state.last_observation:
                ctx.state.add_memory_entry("thought", f"Observation: {ctx.state.last_observation}")
        
        # Prepare request params
        params = {}
        
        # Update settings from strategy
        if ctx.deps.strategy.temperature is not None:
            params['temperature'] = ctx.deps.strategy.temperature
        if ctx.deps.strategy.max_tokens is not None:
            params['max_tokens'] = ctx.deps.strategy.max_tokens
            
        # Prepare prompt with previous observations and thoughts
        memory_summary = ""
        if ctx.state.memory:
            memory_summary = ctx.state.get_memory_summary()
        
        # Prepare the prompt
        prompt = f"{ctx.deps.prompt}\n\n{memory_summary}"
        
        try:
            # First create a ModelSettings object using the tools
            model_settings = None
            if ctx.deps.tools:
                model_settings = ModelSettings(
                    tools=ctx.deps.tools
                )
            
            # Now use the model.request method with the correct parameters
            response, usage = await ctx.deps.model.request(
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                model_settings=model_settings,
                model_request_parameters={
                    "temperature": params.get('temperature', 0.7),
                    "max_tokens": params.get('max_tokens', 2000)
                }
            )
            
            # Add message to history
            if hasattr(response, 'message'):
                ctx.state.message_history.append(response.message)
            
            # Update usage stats
            ctx.state.usage = ctx.deps.run_context.usage
            if hasattr(response, 'usage') and response.usage:
                ctx.state.usage.add(response.usage)
                
            # Check for tool calls first
            if hasattr(response, 'parts'):
                tool_calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
                if tool_calls:
                    # Store information about the tool call
                    tool_name = tool_calls[0].tool_name
                    ctx.state.add_memory_entry("tool_call", f"Called tool: {tool_name}")
                    return ActionNode(tool_calls[0])
                    
                # Then check for text responses
                text_parts = [p for p in response.parts if isinstance(p, TextPart)]
                if text_parts:
                    content = text_parts[0].content
                    
                    # Store thought in memory
                    ctx.state.add_memory_entry("reasoning", content)
                    
                    # Check for final response patterns
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
            
            # If response format is different, try to extract content directly
            if hasattr(response, 'data'):
                content = str(response.data)
                ctx.state.add_memory_entry("reasoning", content)
                return FinalAnswerNode(content)
                    
        except Exception as e:
            # Handle errors gracefully
            error_message = f"Error during reasoning: {str(e)}"
            ctx.state.add_memory_entry("error", error_message)
            
            # Return a final answer with error details to prevent Union instantiation error
            return FinalAnswerNode(f"Final Response: Unable to complete the task due to errors: {error_message}")
            
        # No valid response parts, treat as error
        error_msg = "No valid response from model"
        ctx.state.add_memory_entry("error", error_msg)
        return FinalAnswerNode(f"Final Response: {error_msg}")

@dataclasses.dataclass
class ActionNode(BaseNode):
    """Node for executing actions based on thoughts."""
    tool_call: ToolCallPart
    
    async def run(
        self,
        ctx: GraphRunContext[ReasoningState, ReasoningDeps]
    ) -> GraphResult:  # Use GraphResult instead of Union
        # Get all pending tool calls from the message history
        tool_calls = []
        for message in ctx.state.message_history:
            for part in message.parts:
                if isinstance(part, ToolCallPart) and not part.has_response:
                    tool_calls.append(part)
                    
        # Execute tools sequentially and collect observations
        observations = []
        
        async def execute_tool(tool_call: ToolCallPart) -> tuple[str, ToolCallPart]:
            """Execute a tool and return observation."""
            try:
                # Get tool
                tool = ctx.deps.function_tools.get(tool_call.tool_name)
                if not tool:
                    error_msg = f"Unknown tool: {tool_call.tool_name}"
                    ctx.state.add_memory_entry("error", error_msg)
                    return error_msg, tool_call
                    
                # Parse arguments
                kwargs = {}
                for arg_name, arg_value in tool_call.args.items():
                    kwargs[arg_name] = arg_value
                    
                # Run tool
                result = await tool.run(ctx.deps.run_context, **kwargs)
                
                # Store result in intermediate results
                ctx.state.intermediate_results.append(result)
                
                # Store observation
                observation = str(result)
                ctx.state.add_memory_entry("tool_result", {
                    "tool": tool_call.tool_name,
                    "args": tool_call.args,
                    "result": str(result)[:200] + ("..." if len(str(result)) > 200 else "")
                })
                
                # Mark tool call as responded
                tool_call.response = observation
                tool_call.has_response = True
                
                return observation, tool_call
                
            except Exception as e:
                # Handle tool execution errors
                error_msg = f"Error executing {tool_call.tool_name}: {str(e)}"
                ctx.state.add_memory_entry("error", error_msg)
                
                # Mark tool call as responded with error
                tool_call.response = error_msg
                tool_call.has_response = True
                
                return error_msg, tool_call
                
        # Execute tools
        for tool_call in tool_calls:
            observation, _ = await execute_tool(tool_call)
            observations.append(observation)
            
        # Combine observations
        combined_observation = "\n".join(observations)
        return ObservationNode(combined_observation)

@dataclasses.dataclass
class ObservationNode(BaseNode):
    """Take observation and decide whether to continue or end."""
    observation: str = ""
    
    async def run(
        self,
        ctx: GraphRunContext[ReasoningState, ReasoningDeps]
    ) -> GraphResult:  # Use GraphResult instead of Union
        """Process observation and decide whether to continue."""
        ctx.state.last_observation = self.observation
        ctx.state.loop_count += 1
        ctx.state.add_memory_entry("observation", self.observation)
        
        # If an error is detected in the observation, store it but continue processing
        if "error" in self.observation.lower() and any(e in self.observation.lower() for e in ["exception", "traceback", "failed"]):
            ctx.state.add_memory_entry("error", self.observation)
        
        # Check loop count
        if ctx.state.loop_count >= ctx.deps.strategy.react_config.max_loops:
            # Create a comprehensive Final Response that includes all intermediate results
            if ctx.state.intermediate_results:
                final_message = (
                    f"Reached maximum number of steps ({ctx.deps.strategy.react_config.max_loops}). " +
                    f"Results so far: {ctx.state.intermediate_results}. " +
                    f"Latest observation: {self.observation}"
                )
            else:
                final_message = (
                    f"Reached maximum number of steps ({ctx.deps.strategy.react_config.max_loops}). " +
                    f"Current observation: {self.observation}. " +
                    "Based on the steps taken so far, this appears to be the answer."
                )
            
            # Set the final answer in the state
            ctx.state.final_answer = final_message
            
            # Return a ThoughtNode to FinalAnswerNode which will return the final answer
            # This avoids returning End directly
            node = ThoughtNode()
            node.final_answer = True  # Mark this as a final answer for handling in the ThoughtNode
            return node
            
        return ThoughtNode()

@dataclasses.dataclass
class FinalAnswerNode(BaseNode):
    """Node for producing the Final Response."""
    thought: str
    
    async def run(
        self,
        ctx: GraphRunContext[ReasoningState, ReasoningDeps]
    ) -> GraphResult:  # Use GraphResult instead of Union
        # Extract Final Response from thought
        ctx.state.add_memory_entry("final_answer", self.thought)
        
        # Process the thought to extract the final answer
        if self.thought.startswith("Final Response:"):
            final_answer = self.thought.replace("Final Response:", "").strip()
        elif "Final Response:" in self.thought:
            final_answer = self.thought.split("Final Response:")[1].strip()
        else:
            # If no "Final Response:" prefix, use the entire thought as the answer
            final_answer = self.thought.strip()
            
        # Record the final answer in the state
        ctx.state.final_answer = final_answer
        
        # Add one final memory entry
        ctx.state.add_memory_entry("completion", "Task completed with final answer.")
        
        return End(final_answer)

def build_react_graph() -> Graph:
    """Build the ReAct reasoning graph."""
    # Simple graph builder that avoids Union type instantiation issues
    nodes = (ThoughtNode, ActionNode, ObservationNode, FinalAnswerNode)
    graph = Graph(
        nodes=nodes,
        name="ReAct"
    )
    return graph

async def execute_simple_react(
    prompt: str,
    model: str,
    tools: List[Any] = None,
    run_context: RunContext = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    max_loops: int = 5
) -> str:
    """
    Execute a ReAct reasoning process using the OpenAI API directly.
    
    Args:
        prompt: The user prompt to start the reasoning process
        model: The model to use for reasoning
        tools: The tools available for the agent to use
        run_context: The run context to pass to tools
        temperature: The temperature to use for sampling
        max_tokens: The maximum number of tokens to generate
        max_loops: The maximum number of reasoning loops to execute
    
    Returns:
        The final response from the reasoning process
    """
    try:
        # Prepare the prompt with ReAct instructions
        react_prompt = f"""You are an AI assistant that carefully follows a step-by-step reasoning process:
        1. First, think through the problem step-by-step
        2. If you need to gather information or perform a computation, use one of the available tools
        3. Always show your work and explain your reasoning
        4. When you have a final answer, format it as: FINAL ANSWER: <your answer>
        
        User query: {prompt}
        
        Let's start reasoning step-by-step:
        """
        
        # Prepare the tools
        tool_dict = {tool.name: tool for tool in tools} if tools else {}
        
        # Import necessary classes for OpenAI API
        import openai
        from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
        import importlib
        import inspect
        import pydantic
        
        # Setup the OpenAI client
        client = openai.OpenAI()
        
        # Prepare conversation history
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that can use tools."},
            {"role": "user", "content": react_prompt}
        ]
        
        # Prepare OpenAI tools format
        openai_tools = []
        if tools:
            for tool in tools:
                tool_schema = {}
                # Get tool definition
                if hasattr(tool, 'prepare_tool_def'):
                    tool_def = await tool.prepare_tool_def(run_context)
                    if tool_def:
                        tool_schema = {
                            "type": "function",
                            "function": {
                                "name": tool_def.name,
                                "description": tool_def.description,
                                "parameters": tool_def.parameters_json_schema
                            }
                        }
                        openai_tools.append(tool_schema)

        # Cache for parameter type annotations
        param_types_cache = {}
        
        # Helper function to convert dict to proper parameter types
        async def convert_args_to_proper_types(tool, args_dict):
            """Convert arguments to their proper types based on function signature."""
            # Get function signature
            if not hasattr(tool, 'function'):
                return args_dict
                
            # Check if we've already cached the parameter types for this tool
            if tool.name in param_types_cache:
                param_types = param_types_cache[tool.name]
            else:
                # Analyze the function signature to get parameter types
                signature = inspect.signature(tool.function)
                param_types = {}
                for name, param in signature.parameters.items():
                    # Skip first parameter (context)
                    if name == 'ctx' or name == 'context' or name == 'run_context':
                        continue
                    
                    # Get parameter type
                    if param.annotation != inspect.Parameter.empty:
                        param_types[name] = param.annotation
                
                # Cache the parameter types for this tool
                param_types_cache[tool.name] = param_types
            
            # Convert arguments to their proper types
            converted_args = {}
            for name, value in args_dict.items():
                if name in param_types:
                    param_type = param_types[name]
                    
                    # Handle Pydantic models specially
                    if hasattr(param_type, '__origin__') and param_type.__origin__ is list:
                        # Handle List[X] types
                        item_type = param_type.__args__[0]
                        if issubclass(item_type, pydantic.BaseModel):
                            converted_args[name] = [item_type(**item) for item in value]
                        else:
                            converted_args[name] = value
                    elif isinstance(value, dict) and inspect.isclass(param_type) and issubclass(param_type, pydantic.BaseModel):
                        # Convert dict to Pydantic model
                        logging.info(f"Converting dict to {param_type.__name__} for parameter {name}")
                        converted_args[name] = param_type(**value)
                    else:
                        converted_args[name] = value
                else:
                    converted_args[name] = value
            
            return converted_args
        
        # Main reasoning loop
        for loop_idx in range(max_loops):
            logging.info(f"Starting reasoning loop {loop_idx+1}")
            
            # Call the model
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Get the response content
            response_message = response.choices[0].message
            
            # Add the model's message to the history
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": response_message.tool_calls if hasattr(response_message, 'tool_calls') else None
            })
            
            # Check if we have a final answer
            if response_message.content and "FINAL ANSWER:" in response_message.content:
                # Extract the final answer
                final_answer = response_message.content.split("FINAL ANSWER:")[1].strip()
                return f"Final response: {final_answer}"
            
            # Handle tool calls
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    logging.info(f"Executing tool {tool_name} with args: {tool_args}")
                    
                    if tool_name in tool_dict:
                        try:
                            tool = tool_dict[tool_name]
                            
                            # Convert arguments to proper types
                            converted_args = await convert_args_to_proper_types(tool, tool_args)
                            
                            # Call the tool directly - the tool itself expects run_context as first arg
                            result = await tool.function(run_context, **converted_args)
                            logging.info(f"Tool result: {result}")
                            
                            # Add tool response to the conversation
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(result)
                            })
                        except Exception as e:
                            error_msg = f"Error executing tool {tool_name}: {str(e)}"
                            logging.error(error_msg)
                            
                            # Add error response to the conversation
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_msg
                            })
                    else:
                        # Tool not found
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Tool {tool_name} not found."
                        })
        
        return f"Reached maximum reasoning steps ({max_loops}) without conclusion."
    
    except Exception as e:
        import traceback
        logging.error(f"Error in execute_simple_react: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error during reasoning: {str(e)}"

async def execute_react_reasoning_sync(
    prompt: str,
    model: models.Model,
    tools: List[Tool],
    strategy: ReasoningStrategy,
    run_context: RunContext,
) -> str:
    """Run ReAct reasoning and return the final answer."""
    # Use the simple implementation instead
    return await execute_simple_react(
        prompt=prompt,
        model=model.model_name,
        tools=tools,
        run_context=run_context
    )

async def execute_react_reasoning_stream(
    prompt: str,
    model: models.Model,
    tools: List[Tool],
    strategy: ReasoningStrategy,
    run_context: RunContext,
) -> AsyncIterator[str]:
    """Run ReAct reasoning and stream intermediate results."""
    # For now, just use the sync version and yield the result
    yield "Thinking...\n"
    result = await execute_simple_react(
        prompt=prompt,
        model=model.model_name,
        tools=tools,
        run_context=run_context
    )
    yield result

async def execute_react_reasoning(
    prompt: str,
    model: models.Model,
    tools: List[Tool],
    strategy: ReasoningStrategy,
    run_context: RunContext,
    stream: bool = False
) -> Any:
    """Execute ReAct reasoning strategy, choosing between sync and stream implementations."""
    if stream:
        return execute_react_reasoning_stream(prompt, model, tools, strategy, run_context)
    else:
        return await execute_react_reasoning_sync(prompt, model, tools, strategy, run_context)
