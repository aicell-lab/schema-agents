#!/usr/bin/env python
# coding=utf-8

import asyncio
import sys
from src.schema_agents.local_python_executor import LocalPythonExecutor

async def main():
    print("Initializing Python Executor...")
    executor = LocalPythonExecutor()
    
    print("Running a simple test code...")
    test_code = """
# Simple calculation
a = 10
b = 20
result = a + b
print(f"Result: {result}")
result
"""
    
    try:
        output, logs, _ = await executor(test_code)
        print("\nExecution successful!")
        print(f"Logs:\n{logs}")
        print(f"Output: {output}")
    except Exception as e:
        print(f"Execution failed: {e}")
    finally:
        executor.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 