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
from datetime import datetime

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

async def execute_structured_reasoning(
    prompt: str,
    model: models.Model,
    tools: List[Tool],
    strategy: ReasoningStrategy,
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
        
        # Create initial state
        state = {
            "message_history": [],
            "usage": Usage(),
            "step_count": 0,
            "current_plan": None,
            "episodic_memory": [],
            "final_message": None
        }
        
        # Add initial memory entry
        state["episodic_memory"].append({
            "type": "prompt",
            "content": prompt,
            "step": state["step_count"],
            "timestamp": import_time()
        })
        
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
                memory_summary = ""
                if state["episodic_memory"]:
                    memory_entries = [f"Step {e['step']}: {e['type']} - {str(e['content'])[:100]}..." for e in state["episodic_memory"]]
                    memory_summary = "Previous steps:\n" + "\n".join(memory_entries)
                
                # Include current plan if available
                plan_text = ""
                if state["current_plan"]:
                    plan_text = "Current Plan:\n"
                    for i, step in enumerate(state["current_plan"], 1):
                        plan_text += f"{i}. {step}\n"
                
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
                        openai_tool = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters
                            }
                        }
                        openai_tools.append(openai_tool)
                
                # Request from the model
                chunks.append(f"Thinking (step {current_step + 1})...\n")
                
                response, usage = await model.request(
                    messages=[
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
                    state["usage"].add(response.usage)
                
                # Extract the structured reasoning result
                structured_result = None
                if hasattr(response, 'data') and isinstance(response.data, dict):
                    # Parse the result
                    structured_result = StructuredReasoningResult(**response.data)
                    
                    # Store the thought in memory
                    state["episodic_memory"].append({
                        "type": "thought",
                        "content": structured_result.thought,
                        "step": current_step,
                        "timestamp": import_time()
                    })
                    
                    # Update the plan if provided
                    if structured_result.plan:
                        state["current_plan"] = structured_result.plan
                        state["episodic_memory"].append({
                            "type": "plan",
                            "content": structured_result.plan,
                            "step": current_step,
                            "timestamp": import_time()
                        })
                    
                    # Yield the thought
                    chunks.append(f"Thought: {structured_result.thought}\n")
                    
                    # Check if we have an action to execute
                    if structured_result.action:
                        state["episodic_memory"].append({
                            "type": "action",
                            "content": structured_result.action,
                            "step": current_step,
                            "timestamp": import_time()
                        })
                        
                        # Execute the script
                        chunks.append(f"Executing script...\n")
                        
                        try:
                            # Create a temporary file for the script
                            import tempfile
                            import os
                            import asyncio
                            import sys
                            
                            # Create a temporary file
                            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
                                f.write(structured_result.action)
                                script_path = f.name
                            
                            # Execute the script with timeout
                            timeout = strategy.structured_config.script_timeout
                            
                            try:
                                # Prepare the command
                                cmd = [sys.executable, script_path]
                                
                                # Create subprocess
                                proc = await asyncio.create_subprocess_exec(
                                    *cmd,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE
                                )
                                
                                # Wait for the process with timeout
                                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                                
                                # Decode the output
                                stdout_str = stdout.decode('utf-8')
                                stderr_str = stderr.decode('utf-8')
                                
                                # Prepare the result
                                script_result = f"Script execution result:\n"
                                if stdout_str:
                                    script_result += f"Output:\n{stdout_str}\n"
                                if stderr_str:
                                    script_result += f"Errors:\n{stderr_str}\n"
                                
                                # Clean up the temporary file
                                os.unlink(script_path)
                            except asyncio.TimeoutError:
                                # Kill the process if it times out
                                if 'proc' in locals():
                                    proc.kill()
                                
                                # Clean up the temporary file
                                os.unlink(script_path)
                                
                                # Set the timeout error
                                script_result = f"Script execution timed out after {timeout} seconds"
                            
                            # Store the result in memory
                            state["episodic_memory"].append({
                                "type": "script_result",
                                "content": script_result,
                                "step": current_step,
                                "timestamp": import_time()
                            })
                            
                            # Yield the script result
                            chunks.append(f"Script result: {script_result}\n")
                            
                        except Exception as e:
                            # Handle errors gracefully
                            error_message = f"Error executing script: {str(e)}"
                            state["episodic_memory"].append({
                                "type": "script_error",
                                "content": error_message,
                                "step": current_step,
                                "timestamp": import_time()
                            })
                            
                            # Yield the error
                            chunks.append(f"Error: {error_message}\n")
                    
                    # Yield the message
                    chunks.append(f"Message: {structured_result.message}\n")
                    
                    # Store the message in memory
                    state["episodic_memory"].append({
                        "type": "message",
                        "content": structured_result.message,
                        "step": current_step,
                        "timestamp": import_time()
                    })
                    
                    # Check if we're done
                    if not structured_result.action and (not state["current_plan"] or current_step >= max_steps - 1):
                        # Set the final message
                        state["final_message"] = structured_result.message
                        break
                else:
                    # No valid response
                    error_message = "No valid response from model"
                    state["episodic_memory"].append({
                        "type": "error",
                        "content": error_message,
                        "step": current_step,
                        "timestamp": import_time()
                    })
                    
                    # Yield the error
                    chunks.append(f"Error: {error_message}\n")
                    break
                
                # Increment step count
                current_step += 1
                state["step_count"] = current_step
                
                # Check if we've reached the maximum steps
                if current_step >= max_steps:
                    # Set a final message if not already set
                    if not state["final_message"]:
                        state["final_message"] = f"Reached maximum number of steps ({max_steps})"
                    break
            
            # Check if we need to summarize memory
            if strategy.structured_config.summarize_memory and strategy.structured_config.memory_enabled:
                try:
                    # Generate a summary of the episodic memory
                    memory_entries = [f"Step {e['step']}: {e['type']} - {str(e['content'])}" for e in state["episodic_memory"]]
                    memory_text = "\n".join(memory_entries)
                    
                    # Create a prompt for summarization
                    summary_prompt = f"""Please summarize the following reasoning steps:

{memory_text}

Provide a concise summary of the reasoning process, key insights, and conclusions."""
                    
                    # Request a summary from the model
                    chunks.append(f"Generating summary...\n")
                    
                    response, _ = await model.request(
                        messages=[{
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
                        final_message = state["final_message"] or ""
                        state["final_message"] = f"{final_message}\n\nSummary of reasoning:\n{summary}"
                        
                        # Yield the summary
                        chunks.append(f"Summary: {summary}\n")
                
                except Exception as e:
                    # If summarization fails, just continue with the original message
                    logger.error(f"Error generating memory summary: {str(e)}")
            
            # Yield the final message
            final_message = state["final_message"] or "Reasoning completed"
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
                memory_summary = ""
                if state["episodic_memory"]:
                    memory_entries = [f"Step {e['step']}: {e['type']} - {str(e['content'])[:100]}..." for e in state["episodic_memory"]]
                    memory_summary = "Previous steps:\n" + "\n".join(memory_entries)
                
                # Include current plan if available
                plan_text = ""
                if state["current_plan"]:
                    plan_text = "Current Plan:\n"
                    for i, step in enumerate(state["current_plan"], 1):
                        plan_text += f"{i}. {step}\n"
                
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
                        openai_tool = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters
                            }
                        }
                        openai_tools.append(openai_tool)
                
                # Request from the model
                response, usage = await model.request(
                    messages=[
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
                    state["usage"].add(response.usage)
                
                # Extract the structured reasoning result
                structured_result = None
                if hasattr(response, 'data') and isinstance(response.data, dict):
                    # Parse the result
                    structured_result = StructuredReasoningResult(**response.data)
                    
                    # Store the thought in memory
                    state["episodic_memory"].append({
                        "type": "thought",
                        "content": structured_result.thought,
                        "step": current_step,
                        "timestamp": import_time()
                    })
                    
                    # Update the plan if provided
                    if structured_result.plan:
                        state["current_plan"] = structured_result.plan
                        state["episodic_memory"].append({
                            "type": "plan",
                            "content": structured_result.plan,
                            "step": current_step,
                            "timestamp": import_time()
                        })
                    
                    # Check if we have an action to execute
                    if structured_result.action:
                        state["episodic_memory"].append({
                            "type": "action",
                            "content": structured_result.action,
                            "step": current_step,
                            "timestamp": import_time()
                        })
                        
                        try:
                            # Create a temporary file for the script
                            import tempfile
                            import os
                            import asyncio
                            import sys
                            
                            # Create a temporary file
                            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
                                f.write(structured_result.action)
                                script_path = f.name
                            
                            # Execute the script with timeout
                            timeout = strategy.structured_config.script_timeout
                            
                            try:
                                # Prepare the command
                                cmd = [sys.executable, script_path]
                                
                                # Create subprocess
                                proc = await asyncio.create_subprocess_exec(
                                    *cmd,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE
                                )
                                
                                # Wait for the process with timeout
                                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                                
                                # Decode the output
                                stdout_str = stdout.decode('utf-8')
                                stderr_str = stderr.decode('utf-8')
                                
                                # Prepare the result
                                script_result = f"Script execution result:\n"
                                if stdout_str:
                                    script_result += f"Output:\n{stdout_str}\n"
                                if stderr_str:
                                    script_result += f"Errors:\n{stderr_str}\n"
                                
                                # Clean up the temporary file
                                os.unlink(script_path)
                            except asyncio.TimeoutError:
                                # Kill the process if it times out
                                if 'proc' in locals():
                                    proc.kill()
                                
                                # Clean up the temporary file
                                os.unlink(script_path)
                                
                                # Set the timeout error
                                script_result = f"Script execution timed out after {timeout} seconds"
                            
                            # Store the result in memory
                            state["episodic_memory"].append({
                                "type": "script_result",
                                "content": script_result,
                                "step": current_step,
                                "timestamp": import_time()
                            })
                            
                        except Exception as e:
                            # Handle errors gracefully
                            error_message = f"Error executing script: {str(e)}"
                            state["episodic_memory"].append({
                                "type": "script_error",
                                "content": error_message,
                                "step": current_step,
                                "timestamp": import_time()
                            })
                    
                    # Store the message in memory
                    state["episodic_memory"].append({
                        "type": "message",
                        "content": structured_result.message,
                        "step": current_step,
                        "timestamp": import_time()
                    })
                    
                    # Check if we're done
                    if not structured_result.action and (not state["current_plan"] or current_step >= max_steps - 1):
                        # Set the final message
                        state["final_message"] = structured_result.message
                        break
                else:
                    # No valid response
                    error_message = "No valid response from model"
                    state["episodic_memory"].append({
                        "type": "error",
                        "content": error_message,
                        "step": current_step,
                        "timestamp": import_time()
                    })
                    
                    # Set the final message
                    state["final_message"] = error_message
                    break
                
                # Increment step count
                current_step += 1
                state["step_count"] = current_step
                
                # Check if we've reached the maximum steps
                if current_step >= max_steps:
                    # Set a final message if not already set
                    if not state["final_message"]:
                        state["final_message"] = f"Reached maximum number of steps ({max_steps})"
                    break
            
            # Check if we need to summarize memory
            if strategy.structured_config.summarize_memory and strategy.structured_config.memory_enabled:
                try:
                    # Generate a summary of the episodic memory
                    memory_entries = [f"Step {e['step']}: {e['type']} - {str(e['content'])}" for e in state["episodic_memory"]]
                    memory_text = "\n".join(memory_entries)
                    
                    # Create a prompt for summarization
                    summary_prompt = f"""Please summarize the following reasoning steps:

{memory_text}

Provide a concise summary of the reasoning process, key insights, and conclusions."""
                    
                    # Request a summary from the model
                    response, _ = await model.request(
                        messages=[{
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
                        final_message = state["final_message"] or ""
                        state["final_message"] = f"{final_message}\n\nSummary of reasoning:\n{summary}"
                
                except Exception as e:
                    # If summarization fails, just continue with the original message
                    logger.error(f"Error generating memory summary: {str(e)}")
            
            # Return the final message
            return state["final_message"] or "Reasoning completed"
            
    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error during structured reasoning: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        return error_message
