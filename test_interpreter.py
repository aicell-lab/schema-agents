#!/usr/bin/env python
import os
import logging
from schema_agents.tools.code_interpreter import CodeInterpreter, extract_stdout

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Testing CodeInterpreter...")
    
    # Create work directory
    work_dir_root = "./.data"
    os.makedirs(work_dir_root, exist_ok=True)
    
    try:
        # Try each available kernel
        for kernel_name in ['jupyterlab', 'imjoy-iteractive-ml', 'python27', 'imjoyrpc']:
            logger.info(f"Trying kernel: {kernel_name}")
            try:
                # Create code interpreter
                ci = CodeInterpreter(
                    session_id=f"test_{kernel_name}",
                    work_dir_root=work_dir_root,
                    kernel_name=kernel_name
                )
                
                # Test execution
                logger.info(f"Testing execution with kernel: {kernel_name}")
                result = ci.execute_code("print('Hello from " + kernel_name + "')")
                stdout = extract_stdout(result)
                logger.info(f"Execution result: {result['status']}")
                logger.info(f"Stdout: {stdout}")
                
                # Clean up
                logger.info(f"Tearing down kernel: {kernel_name}")
                ci.tearDown()
                logger.info(f"Successfully tested kernel: {kernel_name}")
                
                # If we got here, we found a working kernel
                break
                
            except Exception as e:
                logger.error(f"Error with kernel {kernel_name}: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 