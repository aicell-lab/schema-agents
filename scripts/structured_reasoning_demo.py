"""
Structured Reasoning Demo

This script demonstrates the structured reasoning approach using the OpenAI API directly,
without relying on the schema-agents framework.

The structured reasoning approach involves:
1. A thought process - The AI's internal reasoning about the problem
2. An optional Python script to execute - Code that can be run to perform calculations or operations
3. An optional plan or revision to the plan - Steps to follow to solve the problem
4. A message to the user - The final output or explanation

Benefits of Structured Reasoning:
1. Transparency: The reasoning process is explicit and visible to the user
2. Verifiability: The AI's thought process can be examined and verified
3. Executability: The AI can write and execute code to perform calculations
4. Planning: The AI can create and revise plans to solve complex problems
5. Episodic Memory: The AI maintains a history of its reasoning steps
6. Summarization: The AI can summarize its reasoning process at the end

This approach is particularly useful for:
- Complex problem-solving that requires multiple steps
- Tasks that benefit from executing code (calculations, data analysis)
- Explanations that need to show the reasoning process
- Educational contexts where showing the work is important

The implementation uses:
- Pydantic for structured data validation
- AsyncOpenAI for API calls
- JSON response format for structured outputs
- Asyncio for asynchronous execution
- Temporary files for script execution

Example prompts included:
1. Calculate the area of a circle with radius 5 cm
2. Analyze the Fibonacci sequence and write a script to calculate the first 10 Fibonacci numbers
3. Explain the concept of recursion and provide a simple example
"""

import os
import json
import asyncio
import tempfile
import subprocess
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Ensure OPENAI_API_KEY is set
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Define the structured reasoning result model
class StructuredReasoningResult(BaseModel):
    """Result model for structured reasoning."""
    thought: str = Field(..., description="The reasoning process")
    action: Optional[str] = Field(None, description="Python script to be executed")
    plan: Optional[List[str]] = Field(None, description="List of steps to follow")
    message: str = Field(..., description="Message to the user")

async def execute_script(script_content: str, timeout: int = 30) -> str:
    """Execute a Python script and return the result."""
    try:
        # Create a temporary file for the script
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(script_content)
            script_path = f.name
        
        # Execute the script with timeout
        try:
            # Prepare the command
            cmd = ["python", script_path]
            
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
            
            return script_result
            
        except asyncio.TimeoutError:
            # Kill the process if it times out
            if 'proc' in locals():
                proc.kill()
            
            # Clean up the temporary file
            os.unlink(script_path)
            
            # Set the timeout error
            return f"Script execution timed out after {timeout} seconds"
            
    except Exception as e:
        # Handle errors gracefully
        return f"Error executing script: {str(e)}"

async def structured_reasoning(prompt: str, max_steps: int = 5) -> None:
    """Execute structured reasoning using the OpenAI API directly."""
    # Initialize the OpenAI client
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Initialize state
    state = {
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
        "timestamp": datetime.now().isoformat()
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
    
    # Main reasoning loop
    current_step = 0
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
        
        print(f"\n--- Step {current_step + 1} ---")
        print("Thinking...")
        
        # Request from the model
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract the structured reasoning result
        try:
            result_json = json.loads(response.choices[0].message.content)
            structured_result = StructuredReasoningResult(**result_json)
            
            # Store the thought in memory
            state["episodic_memory"].append({
                "type": "thought",
                "content": structured_result.thought,
                "step": current_step,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update the plan if provided
            if structured_result.plan:
                state["current_plan"] = structured_result.plan
                state["episodic_memory"].append({
                    "type": "plan",
                    "content": structured_result.plan,
                    "step": current_step,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Print the thought
            print(f"\nThought: {structured_result.thought}")
            
            # Check if we have an action to execute
            if structured_result.action:
                state["episodic_memory"].append({
                    "type": "action",
                    "content": structured_result.action,
                    "step": current_step,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Execute the script
                print("\nExecuting script...")
                script_result = await execute_script(structured_result.action)
                
                # Store the result in memory
                state["episodic_memory"].append({
                    "type": "script_result",
                    "content": script_result,
                    "step": current_step,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Print the script result
                print(f"\nScript result: {script_result}")
            
            # Print the message
            print(f"\nMessage: {structured_result.message}")
            
            # Store the message in memory
            state["episodic_memory"].append({
                "type": "message",
                "content": structured_result.message,
                "step": current_step,
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if we're done
            if not structured_result.action and (not state["current_plan"] or current_step >= max_steps - 1):
                # Set the final message
                state["final_message"] = structured_result.message
                break
                
        except Exception as e:
            # Handle errors gracefully
            error_message = f"Error processing model response: {str(e)}"
            state["episodic_memory"].append({
                "type": "error",
                "content": error_message,
                "step": current_step,
                "timestamp": datetime.now().isoformat()
            })
            
            # Print the error
            print(f"\nError: {error_message}")
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
    
    # Generate a summary of the reasoning process
    print("\n--- Summary ---")
    
    # Request a summary from the model
    summary_prompt = f"""Please summarize the following reasoning steps:

{memory_summary}

Provide a concise summary of the reasoning process, key insights, and conclusions."""
    
    # Request a summary from the model
    summary_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3,
        max_tokens=500
    )
    
    # Print the summary
    print(f"\nSummary: {summary_response.choices[0].message.content}")
    
    # Print the final message
    print(f"\nFinal result: {state['final_message'] or 'Reasoning completed'}")

async def main():
    """Main function to run the structured reasoning demo."""
    # Example prompts
    prompts = [
        "Calculate the area of a circle with radius 5 cm.",
        "Analyze the Fibonacci sequence and write a script to calculate the first 10 Fibonacci numbers.",
        "Explain the concept of recursion and provide a simple example."
    ]
    
    # Select a prompt
    selected_prompt = prompts[0]  # Using the circle area calculation example
    
    print(f"Running structured reasoning with prompt: {selected_prompt}")
    await structured_reasoning(selected_prompt)

if __name__ == "__main__":
    asyncio.run(main()) 