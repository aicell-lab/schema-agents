#!/usr/bin/env python
"""
Test script for structured reasoning with CodeInterpreter integration.
"""

import asyncio
import os
import sys
import logging
from pydantic import BaseModel, Field
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import schema_agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules we need to test
from schema_agents.structured_reasoning import (
    execute_structured_reasoning,
    StructuredReasoningResult,
    get_code_interpreter,
    cleanup_code_interpreter
)

# Define a simple reasoning strategy for testing
class StructuredConfig(BaseModel):
    """Configuration for structured reasoning."""
    max_steps: int = Field(3, description="Maximum number of reasoning steps")
    script_timeout: int = Field(30, description="Timeout for script execution in seconds")
    memory_enabled: bool = Field(True, description="Whether to enable episodic memory")
    summarize_memory: bool = Field(True, description="Whether to summarize memory at the end")

class ReasoningStrategy(BaseModel):
    """Strategy for structured reasoning."""
    temperature: float = Field(0.7, description="Temperature for model generation")
    max_tokens: int = Field(2000, description="Maximum tokens for model generation")
    structured_config: StructuredConfig = Field(default_factory=StructuredConfig, description="Configuration for structured reasoning")

# Mock model for testing
class MockModel:
    """Mock model for testing."""
    
    async def request(self, message_history, model_settings=None, model_request_parameters=None):
        """Mock request method."""
        # Return a simple response with a Python script to execute
        response = type('Response', (), {})()
        response.usage = None
        response.data = {
            "thought": "I'll create a simple Python script to test the integration.",
            "action": """
import os
import sys
import platform

# Print some system information
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"Current directory: {os.getcwd()}")

# Create a simple calculation
result = 0
for i in range(10):
    result += i
print(f"Sum of numbers from 0 to 9: {result}")

# Test error handling
try:
    # This will cause a ZeroDivisionError
    x = 1 / 0
except Exception as e:
    print(f"Caught exception: {type(e).__name__}: {str(e)}")
""",
            "plan": ["Step 1: Create a test script", "Step 2: Execute the script", "Step 3: Verify the results"],
            "message": "I've created a simple Python script to test the integration."
        }
        return response, None

async def test_structured_reasoning():
    """Test the structured reasoning with CodeInterpreter integration."""
    try:
        # Create a test directory
        test_dir = "./.test-structured-reasoning"
        os.makedirs(test_dir, exist_ok=True)
        
        # Initialize the code interpreter
        code_interpreter = get_code_interpreter(work_dir_root=test_dir)
        
        # Define a simple prompt
        prompt = "Create a simple Python script to test the integration."
        
        # Create a mock model
        model = MockModel()
        
        # Define a simple strategy
        strategy = ReasoningStrategy()
        
        # Execute the structured reasoning
        logger.info("Starting structured reasoning test...")
        result = await execute_structured_reasoning(
            prompt=prompt,
            model=model,
            tools=[],
            strategy=strategy,
            run_context=None,
            stream=False
        )
        
        logger.info(f"Structured reasoning result: {result}")
        
        # Test streaming mode
        logger.info("Testing streaming mode...")
        stream_result = await execute_structured_reasoning(
            prompt=prompt,
            model=model,
            tools=[],
            strategy=strategy,
            run_context=None,
            stream=True
        )
        
        # Collect streaming results
        chunks = []
        async for chunk in stream_result:
            chunks.append(chunk)
            logger.info(f"Received chunk: {chunk}")
        
        logger.info(f"Received {len(chunks)} chunks in streaming mode")
        
        # Clean up
        cleanup_code_interpreter()
        
        logger.info("Test completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = asyncio.run(test_structured_reasoning())
    sys.exit(0 if success else 1) 