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
import os
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
from datetime import datetime

# Import CodeInterpreter
from .tools.code_interpreter import CodeInterpreter, extract_stdout, extract_stderr, extract_error

# Configure logging
logger = logging.getLogger("structured_reasoning")
logger.setLevel(logging.INFO)

# Define StructuredReasoningResult class here to avoid circular imports
class StructuredReasoningResult(BaseModel):
    """Result model for structured reasoning."""
    thought: str = Field(..., description="The reasoning process")
    action: Optional[str] = Field(None, description="Python script to be executed")
    plan: Optional[List[str]] = Field(None, description="List of steps to follow")
    message: str = Field(..., description="Message to the user")

# Helper function for timestamp
def import_time() -> str:
    """Import time for timestamping."""
    return datetime.now().isoformat()

# Define Pydantic models for state management
class MemoryEntry(BaseModel):
    """Model for a memory entry in the episodic memory."""
    type: str = Field(..., description="Type of memory entry (prompt, thought, action, etc.)")
    content: Any = Field(..., description="Content of the memory entry")
    step: int = Field(..., description="Step number when this entry was created")
    timestamp: str = Field(default_factory=import_time, description="Timestamp when this entry was created")

class StructuredReasoningState(BaseModel):
    """State model for structured reasoning execution."""
    message_history: List[Any] = Field(default_factory=list, description="History of model messages")
    usage: Usage = Field(default_factory=Usage, description="Usage statistics")
    step_count: int = Field(0, description="Current step count")
    current_plan: Optional[List[str]] = Field(None, description="Current plan steps if available")
    episodic_memory: List[MemoryEntry] = Field(default_factory=list, description="Episodic memory entries")
    final_message: Optional[str] = Field(None, description="Final message to return")
    
    def add_memory_entry(self, entry_type: str, content: Any) -> None:
        """Add an entry to the episodic memory."""
        self.episodic_memory.append(
            MemoryEntry(
                type=entry_type,
                content=content,
                step=self.step_count,
                timestamp=import_time()
            )
        )
    
    def get_memory_summary(self) -> str:
        """Get a summary of the memory for prompt enhancement."""
        if not self.episodic_memory:
            return ""
        
        summary = "Previous steps:\n"
        for entry in self.episodic_memory:
            summary += f"Step {entry.step}: {entry.type} - {str(entry.content)[:100]}...\n"
        return summary
    
    def get_current_plan_text(self) -> str:
        """Get the current plan as formatted text."""
        if not self.current_plan:
            return "No plan has been created yet."
        
        plan_text = "Current Plan:\n"
        for i, step in enumerate(self.current_plan, 1):
            plan_text += f"{i}. {step}\n"
        return plan_text

# Global variable to store the CodeInterpreter instance
_code_interpreter_instance = None

def get_code_interpreter(work_dir_root="./.code-interpreter", reset=False):
    """Get or create a CodeInterpreter instance.
    
    Args:
        work_dir_root: The root directory for code interpreter workspace
        reset: Whether to reset the interpreter if it already exists
        
    Returns:
        A CodeInterpreter instance
    """
    global _code_interpreter_instance
    
    # Create the work directory if it doesn't exist
    os.makedirs(work_dir_root, exist_ok=True)
    
    # Create a new instance if needed
    if _code_interpreter_instance is None:
        logger.info(f"Creating new CodeInterpreter instance with work_dir_root: {work_dir_root}")
        _code_interpreter_instance = CodeInterpreter(work_dir_root=work_dir_root)
    elif reset:
        logger.info("Resetting CodeInterpreter instance")
        _code_interpreter_instance.reset()
    
    return _code_interpreter_instance

async def execute_structured_reasoning(
    prompt: str,
    model: models.Model,
    tools: List[Tool],
    strategy: 'ReasoningStrategy',
    run_context: RunContext,
    stream: bool = False
) -> Any:
    """Execute Structured Reasoning strategy.
    
    Args:
        prompt: The user prompt to start the reasoning process
        model: The model to use for reasoning
        tools: The tools available for the agent to use
        strategy: The reasoning strategy configuration
        run_context: The run context to pass to tools
        stream: Whether to stream the results
        
    Returns:
        The final response from the reasoning process
    """
    try:
        # Prepare function tools
        function_tools = {}
        for tool in tools:
            if hasattr(tool, 'name'):
                function_tools[tool.name] = tool
        
        # Create initial state using Pydantic model
        state = StructuredReasoningState()
        
        # Add initial memory entry
        state.add_memory_entry("prompt", prompt)
        
        # Prepare the system prompt for structured reasoning
        system_prompt = """You are an AI assistant that follows a structured reasoning process.
For each step, you will provide:
1. Your thought process
2. An optional Python script to execute (if needed)
3. An optional plan or revision to the plan
4. A message to the user

Always format your response as a JSON object with the following structure:
{
  "thought": "Your detailed reasoning process",
  "action": "Python script to execute (optional)",
  "plan": ["Step 1", "Step 2", ...] (optional),
  "message": "Your message to the user"
}
"""
        
        if stream:
            # Create a list to collect the stream chunks
            chunks = []
            
            # Add the initial prompt
            chunks.append(f"Starting structured reasoning with prompt: {prompt}\n\n")
            
            # Initialize variables
            current_step = 0
            max_steps = strategy.structured_config.max_steps
            
            while current_step < max_steps:
                # Prepare the prompt with previous observations and thoughts
                memory_summary = state.get_memory_summary()
                
                # Include current plan if available
                plan_text = state.get_current_plan_text()
                
                # Prepare the full prompt
                full_prompt = f"{prompt}\n\n{plan_text}\n\n{memory_summary}"
                
                # Create model settings with the schema
                model_settings = ModelSettings(
                    result_schema={
                        "schema": StructuredReasoningResult.model_json_schema(),
                        "description": "Structured reasoning result with thought, action, plan, and message"
                    }
                )
                
                # Convert tools to OpenAI format
                openai_tools = []
                if tools:
                    for tool in tools:
                        # Get tool definition using prepare_tool_def if available
                        if hasattr(tool, 'prepare_tool_def'):
                            # Create a minimal run context for the tool
                            minimal_ctx = RunContext(
                                deps=None,
                                model=model,
                                usage=Usage(),
                                prompt=prompt,
                                messages=[],
                                run_step=0
                            )
                            
                            # Get tool definition
                            tool_def = await tool.prepare_tool_def(minimal_ctx)
                            if tool_def:
                                openai_tool = {
                                    "type": "function",
                                    "function": {
                                        "name": tool_def.name,
                                        "description": tool_def.description,
                                        "parameters": tool_def.parameters_json_schema
                                    }
                                }
                                openai_tools.append(openai_tool)
                        # Fallback for tools that don't have prepare_tool_def
                        elif hasattr(tool, 'name') and hasattr(tool, 'description'):
                            # Try to extract parameters from the tool's signature
                            parameters = {}
                            if hasattr(tool, 'run') and callable(tool.run):
                                import inspect
                                sig = inspect.signature(tool.run)
                                parameters = {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }
                                
                                # Skip the first parameter (ctx) and extract the rest
                                for i, (param_name, param) in enumerate(list(sig.parameters.items())[1:]):
                                    if param.annotation != inspect.Parameter.empty:
                                        param_type = "string"
                                        if param.annotation == int:
                                            param_type = "integer"
                                        elif param.annotation == float:
                                            param_type = "number"
                                        elif param.annotation == bool:
                                            param_type = "boolean"
                                        
                                        parameters["properties"][param_name] = {
                                            "type": param_type,
                                            "description": f"Parameter {param_name}"
                                        }
                                        
                                        if param.default == inspect.Parameter.empty:
                                            parameters["required"].append(param_name)
                            
                            openai_tool = {
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": parameters
                                }
                            }
                            openai_tools.append(openai_tool)
                
                # Request from the model
                chunks.append(f"Thinking (step {current_step + 1})...\n")
                
                response, usage = await model.request(
                    message_history=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    model_settings=model_settings,
                    model_request_parameters={
                        "temperature": strategy.temperature,
                        "max_tokens": strategy.max_tokens,
                        "tools": openai_tools if openai_tools else None
                    }
                )
                
                # Update usage stats
                if hasattr(response, 'usage') and response.usage:
                    state.usage.add(response.usage)
                
                # Extract the structured reasoning result
                structured_result = None
                if hasattr(response, 'data') and isinstance(response.data, dict):
                    # Parse the result
                    structured_result = StructuredReasoningResult(**response.data)
                    
                    # Store the thought in memory
                    state.add_memory_entry("thought", structured_result.thought)
                    
                    # Update the plan if provided
                    if structured_result.plan:
                        state.current_plan = structured_result.plan
                        state.add_memory_entry("plan", structured_result.plan)
                    
                    # Yield the thought
                    chunks.append(f"Thought: {structured_result.thought}\n")
                    
                    # Check if we have an action to execute
                    if structured_result.action:
                        state.add_memory_entry("action", structured_result.action)
                        
                        # Execute the script
                        chunks.append(f"Executing script...\n")
                        
                        try:
                            # Get the code interpreter
                            code_interpreter = get_code_interpreter()
                            
                            # Execute the code with timeout
                            timeout = strategy.structured_config.script_timeout
                            
                            try:
                                # Execute the code
                                results = code_interpreter.execute_code(
                                    structured_result.action,
                                    timeout=timeout
                                )
                                
                                # Extract the output
                                stdout_str = extract_stdout(results)
                                stderr_str = extract_stderr(results)
                                error_str = extract_error(results)
                                
                                # Prepare the result
                                script_result = f"Script execution result:\n"
                                if stdout_str:
                                    script_result += f"Output:\n{stdout_str}\n"
                                if stderr_str:
                                    script_result += f"Errors:\n{stderr_str}\n"
                                if error_str:
                                    script_result += f"Errors:\n{error_str}\n"
                                
                                # Check execution status
                                if results["status"] != "ok":
                                    script_result += f"Execution failed with status: {results['status']}\n"
                                    if results.get("traceback"):
                                        script_result += f"Traceback:\n{results['traceback']}\n"
                            except Exception as e:
                                # Handle timeout or other errors
                                script_result = f"Script execution error: {str(e)}"
                            
                            # Store the result in memory
                            state.add_memory_entry("script_result", script_result)
                            
                            # Yield the script result
                            chunks.append(f"Script result: {script_result}\n")
                            
                        except Exception as e:
                            # Handle errors gracefully
                            error_message = f"Error executing script: {str(e)}"
                            state.add_memory_entry("script_error", error_message)
                            
                            # Yield the error
                            chunks.append(f"Error: {error_message}\n")
                    
                    # Yield the message
                    chunks.append(f"Message: {structured_result.message}\n")
                    
                    # Store the message in memory
                    state.add_memory_entry("message", structured_result.message)
                    
                    # Check if we're done
                    if not structured_result.action and (not state.current_plan or current_step >= max_steps - 1):
                        # Set the final message
                        state.final_message = structured_result.message
                        break
                else:
                    # No valid response
                    error_message = "No valid response from model"
                    state.add_memory_entry("error", error_message)
                    
                    # Yield the error
                    chunks.append(f"Error: {error_message}\n")
                    break
                
                # Increment step count
                current_step += 1
                state.step_count = current_step
                
                # Check if we've reached the maximum steps
                if current_step >= max_steps:
                    # Set a final message if not already set
                    if not state.final_message:
                        state.final_message = f"Reached maximum number of steps ({max_steps})"
                    break
            
            # Check if we need to summarize memory
            if hasattr(strategy, 'structured_config') and hasattr(strategy.structured_config, 'summarize_memory') and hasattr(strategy.structured_config, 'memory_enabled'):
                if strategy.structured_config.summarize_memory and strategy.structured_config.memory_enabled:
                    try:
                        # Generate a summary of the episodic memory
                        memory_entries = [f"Step {e.step}: {e.type} - {str(e.content)}" for e in state.episodic_memory]
                        memory_text = "\n".join(memory_entries)
                        
                        # Create a prompt for summarization
                        summary_prompt = f"""Please summarize the following reasoning steps:

{memory_text}

Provide a concise summary of the reasoning process, key insights, and conclusions."""
                        
                        # Request a summary from the model
                        chunks.append(f"Generating summary...\n")
                        
                        response, _ = await model.request(
                            message_history=[{
                                "role": "user",
                                "content": summary_prompt
                            }],
                            model_request_parameters={
                                "temperature": 0.3,
                                "max_tokens": 500
                            }
                        )
                        
                        # Extract the summary
                        summary = ""
                        if hasattr(response, 'parts'):
                            text_parts = [p for p in response.parts if isinstance(p, TextPart)]
                            if text_parts:
                                summary = text_parts[0].content
                        elif hasattr(response, 'data'):
                            summary = str(response.data)
                        
                        # Add the summary to the final message
                        if summary:
                            final_message = state.final_message or ""
                            state.final_message = f"{final_message}\n\nSummary of reasoning:\n{summary}"
                            
                            # Yield the summary
                            chunks.append(f"Summary: {summary}\n")
                    
                    except Exception as e:
                        # If summarization fails, just continue with the original message
                        logger.error(f"Error generating memory summary: {str(e)}")
            
            # Yield the final message
            final_message = state.final_message or "Reasoning completed"
            chunks.append(f"Final result: {final_message}\n")
            
            # Create an async generator to yield the chunks
            async def async_generator():
                for chunk in chunks:
                    yield chunk
            
            return async_generator()
        else:
            # Non-streaming implementation
            current_step = 0
            max_steps = strategy.structured_config.max_steps
            
            while current_step < max_steps:
                # Prepare the prompt with previous observations and thoughts
                memory_summary = state.get_memory_summary()
                
                # Include current plan if available
                plan_text = state.get_current_plan_text()
                
                # Prepare the full prompt
                full_prompt = f"{prompt}\n\n{plan_text}\n\n{memory_summary}"
                
                # Create model settings with the schema
                model_settings = ModelSettings(
                    result_schema={
                        "schema": StructuredReasoningResult.model_json_schema(),
                        "description": "Structured reasoning result with thought, action, plan, and message"
                    }
                )
                
                # Convert tools to OpenAI format
                openai_tools = []
                if tools:
                    for tool in tools:
                        # Get tool definition using prepare_tool_def if available
                        if hasattr(tool, 'prepare_tool_def'):
                            # Create a minimal run context for the tool
                            minimal_ctx = RunContext(
                                deps=None,
                                model=model,
                                usage=Usage(),
                                prompt=prompt,
                                messages=[],
                                run_step=0
                            )
                            
                            # Get tool definition
                            tool_def = await tool.prepare_tool_def(minimal_ctx)
                            if tool_def:
                                openai_tool = {
                                    "type": "function",
                                    "function": {
                                        "name": tool_def.name,
                                        "description": tool_def.description,
                                        "parameters": tool_def.parameters_json_schema
                                    }
                                }
                                openai_tools.append(openai_tool)
                        # Fallback for tools that don't have prepare_tool_def
                        elif hasattr(tool, 'name') and hasattr(tool, 'description'):
                            # Try to extract parameters from the tool's signature
                            parameters = {}
                            if hasattr(tool, 'run') and callable(tool.run):
                                import inspect
                                sig = inspect.signature(tool.run)
                                parameters = {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }
                                
                                # Skip the first parameter (ctx) and extract the rest
                                for i, (param_name, param) in enumerate(list(sig.parameters.items())[1:]):
                                    if param.annotation != inspect.Parameter.empty:
                                        param_type = "string"
                                        if param.annotation == int:
                                            param_type = "integer"
                                        elif param.annotation == float:
                                            param_type = "number"
                                        elif param.annotation == bool:
                                            param_type = "boolean"
                                        
                                        parameters["properties"][param_name] = {
                                            "type": param_type,
                                            "description": f"Parameter {param_name}"
                                        }
                                        
                                        if param.default == inspect.Parameter.empty:
                                            parameters["required"].append(param_name)
                            
                            openai_tool = {
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": parameters
                                }
                            }
                            openai_tools.append(openai_tool)
                
                # Request from the model
                response, usage = await model.request(
                    message_history=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    model_settings=model_settings,
                    model_request_parameters={
                        "temperature": strategy.temperature,
                        "max_tokens": strategy.max_tokens,
                        "tools": openai_tools if openai_tools else None
                    }
                )
                
                # Update usage stats
                if hasattr(response, 'usage') and response.usage:
                    state.usage.add(response.usage)
                
                # Extract the structured reasoning result
                structured_result = None
                if hasattr(response, 'data') and isinstance(response.data, dict):
                    # Parse the result
                    structured_result = StructuredReasoningResult(**response.data)
                    
                    # Store the thought in memory
                    state.add_memory_entry("thought", structured_result.thought)
                    
                    # Update the plan if provided
                    if structured_result.plan:
                        state.current_plan = structured_result.plan
                        state.add_memory_entry("plan", structured_result.plan)
                    
                    # Check if we have an action to execute
                    if structured_result.action:
                        state.add_memory_entry("action", structured_result.action)
                        
                        try:
                            # Get the code interpreter
                            code_interpreter = get_code_interpreter()
                            
                            # Execute the code with timeout
                            timeout = strategy.structured_config.script_timeout
                            
                            try:
                                # Execute the code
                                results = code_interpreter.execute_code(
                                    structured_result.action,
                                    timeout=timeout
                                )
                                
                                # Extract the output
                                stdout_str = extract_stdout(results)
                                stderr_str = extract_stderr(results)
                                error_str = extract_error(results)
                                
                                # Prepare the result
                                script_result = f"Script execution result:\n"
                                if stdout_str:
                                    script_result += f"Output:\n{stdout_str}\n"
                                if stderr_str:
                                    script_result += f"Errors:\n{stderr_str}\n"
                                if error_str:
                                    script_result += f"Errors:\n{error_str}\n"
                                
                                # Check execution status
                                if results["status"] != "ok":
                                    script_result += f"Execution failed with status: {results['status']}\n"
                                    if results.get("traceback"):
                                        script_result += f"Traceback:\n{results['traceback']}\n"
                            except Exception as e:
                                # Handle timeout or other errors
                                script_result = f"Script execution error: {str(e)}"
                            
                            # Store the result in memory
                            state.add_memory_entry("script_result", script_result)
                            
                        except Exception as e:
                            # Handle errors gracefully
                            error_message = f"Error executing script: {str(e)}"
                            state.add_memory_entry("script_error", error_message)
                    
                    # Store the message in memory
                    state.add_memory_entry("message", structured_result.message)
                    
                    # Check if we're done
                    if not structured_result.action and (not state.current_plan or current_step >= max_steps - 1):
                        # Set the final message
                        state.final_message = structured_result.message
                        break
                else:
                    # No valid response
                    error_message = "No valid response from model"
                    state.add_memory_entry("error", error_message)
                    
                    # Set the final message
                    state.final_message = error_message
                    break
                
                # Increment step count
                current_step += 1
                state.step_count = current_step
                
                # Check if we've reached the maximum steps
                if current_step >= max_steps:
                    # Set a final message if not already set
                    if not state.final_message:
                        state.final_message = f"Reached maximum number of steps ({max_steps})"
                    break
            
            # Check if we need to summarize memory
            if hasattr(strategy, 'structured_config') and hasattr(strategy.structured_config, 'summarize_memory') and hasattr(strategy.structured_config, 'memory_enabled'):
                if strategy.structured_config.summarize_memory and strategy.structured_config.memory_enabled:
                    try:
                        # Generate a summary of the episodic memory
                        memory_entries = [f"Step {e.step}: {e.type} - {str(e.content)}" for e in state.episodic_memory]
                        memory_text = "\n".join(memory_entries)
                        
                        # Create a prompt for summarization
                        summary_prompt = f"""Please summarize the following reasoning steps:

{memory_text}

Provide a concise summary of the reasoning process, key insights, and conclusions."""
                        
                        # Request a summary from the model
                        response, _ = await model.request(
                            message_history=[{
                                "role": "user",
                                "content": summary_prompt
                            }],
                            model_request_parameters={
                                "temperature": 0.3,
                                "max_tokens": 500
                            }
                        )
                        
                        # Extract the summary
                        summary = ""
                        if hasattr(response, 'parts'):
                            text_parts = [p for p in response.parts if isinstance(p, TextPart)]
                            if text_parts:
                                summary = text_parts[0].content
                        elif hasattr(response, 'data'):
                            summary = str(response.data)
                        
                        # Add the summary to the final message
                        if summary:
                            final_message = state.final_message or ""
                            state.final_message = f"{final_message}\n\nSummary of reasoning:\n{summary}"
                    
                    except Exception as e:
                        # If summarization fails, just continue with the original message
                        logger.error(f"Error generating memory summary: {str(e)}")
            
            # Return the final message
            return state.final_message or "Reasoning completed"
            
    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error during structured reasoning: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        return error_message
    finally:
        # Ensure we don't leak resources if we're using a temporary code interpreter
        # We don't shut down the global instance here, as it might be reused
        pass

# Cleanup function to ensure CodeInterpreter is properly shut down
def cleanup_code_interpreter():
    """Clean up the CodeInterpreter instance when the module is unloaded."""
    global _code_interpreter_instance
    if _code_interpreter_instance is not None:
        logger.info("Shutting down CodeInterpreter instance")
        try:
            _code_interpreter_instance.tearDown()
        except Exception as e:
            logger.error(f"Error shutting down CodeInterpreter: {str(e)}")
        _code_interpreter_instance = None

# Register the cleanup function to be called when the module is unloaded
import atexit
atexit.register(cleanup_code_interpreter)
