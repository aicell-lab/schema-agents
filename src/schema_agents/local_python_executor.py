#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import asyncio
import builtins
import concurrent.futures
import difflib
import inspect
import logging
import math
import os
import re
import shortuuid
import base64
import json
from collections.abc import Mapping
from functools import wraps
from importlib import import_module
from importlib.util import find_spec
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Jupyter imports
from jupyter_client.manager import start_new_kernel
from jupyter_client.utils import run_sync
from .msgspec_v5 import validate_message # Assuming msgspec_v5 is available locally

from .tools import Tool
from .utils import BASE_BUILTIN_MODULES, truncate_content, FINAL_ANSWER_MARKER


logger = logging.getLogger(__name__)

# +++ Helper functions from code_interpreter.py +++
def strip_ansi_codes(s):
    """
    Strip ANSI escape codes from a string.
    Handles both actual escape sequences and their string literal representations.
    """
    if not isinstance(s, str):
        return s
        
    # Handle raw ANSI escape sequences
    s = re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', s)
    
    # Handle string literal representations like '\x1b[31m'
    s = re.sub(r'\\x1b\[[0-9;]*[mK]', '', s)
    s = re.sub(r'\\x1B\[[0-9;]*[mK]', '', s)
    
    # Handle other common escape sequence representations
    s = re.sub(r'\\u001b\[[0-9;]*[mK]', '', s)
    s = re.sub(r'\033\[[0-9;]*[mK]', '', s)
    
    # Handle ANSI color codes in the format \x1b[XXm or \x1b[XX;XXm
    s = re.sub(r'\\x1b\[\d+(?:;\d+)*m', '', s)
    
    # Handle ANSI SGR (Select Graphic Rendition) parameters
    s = re.sub(r'\\x1b\[\d+;\d+;\d+m', '', s)
    
    # Handle raw unicode ANSI color codes for terminal colors (38;5;)
    s = re.sub(r'\\x1b\[38;5;\d+(?:;\d+)*m', '', s)
    
    return s

TIMEOUT=60 # Increased timeout for potentially longer operations

INIT_CODE = """%matplotlib inline
%cd {work_dir}
import sys
sys.path.append('{work_dir}')
import os
import math
import builtins # Make basic builtins explicitly available if needed
# Add other common imports if desired
import numpy as np
import pandas as pd
"""

def extract_stdout(results):
    output = results["outputs"]
    return "\\n".join([item['stdout'] for item in output if 'stdout' in item])

def extract_stderr(results):
    output = results["outputs"]
    return "\\n".join([item['stderr'] for item in output if 'stderr' in item])

def extract_error(results):
    output = results["outputs"]
    return "\\n".join([item['error'] for item in output if 'error' in item])

def extract_execute_result(results):
    output = results["outputs"]
    return "\\n".join([item['execute_result'] for item in output if 'execute_result' in item])

def extract_display_data(results):
    output = results["outputs"]
    return "\\n".join([item['display_data'] for item in output if 'display_data' in item])
# --- End Helper functions ---


def fix_final_answer_code(code: str) -> str:
    """
    Sometimes an LLM can try to assign a variable to final_answer, which would break the final_answer() tool.
    This function fixes this behaviour by replacing variable assignments to final_answer with final_answer_variable,
    while preserving function calls to final_answer().
    """
    # First, find if there's a direct assignment to final_answer
    # Use word boundary and negative lookbehind to ensure it's not an object attribute
    assignment_pattern = r"(?<!\.)(?<!\w)\bfinal_answer\s*="
    if "final_answer(" not in code or not re.search(assignment_pattern, code):
        # If final_answer tool is not called in this blob, then doing the replacement is hazardous because it could false the model's memory for next steps.
        # Let's not modify the code and leave the subsequent assignment error happen.
        return code

    # Pattern for replacing variable assignments
    # Looks for 'final_answer' followed by '=' with optional whitespace
    # Negative lookbehind ensures we don't match object attributes
    assignment_regex = r"(?<!\.)(?<!\w)(\bfinal_answer)(\s*=)"
    code = re.sub(assignment_regex, r"final_answer_variable\2", code)

    # Pattern for replacing variable usage but not function calls
    # Negative lookahead (?!\s*\() ensures we don't match function calls
    # Negative lookbehind (?<!\.|\w) ensures we don't match object methods or other variables
    variable_regex = r"(?<!\.)(?<!\w)(\bfinal_answer\b)(?!\s*\()"
    code = re.sub(variable_regex, "final_answer_variable", code)
    return code


class InterpreterError(ValueError):
    """
    An error raised when the interpreter cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    """
    pass


DEFAULT_MAX_LEN_OUTPUT = 50000


class PythonExecutor:
    pass


class LocalPythonExecutor(PythonExecutor):
    def __init__(
        self,
        additional_authorized_imports: List[str] = None,
        max_print_outputs_length: Optional[int] = None,
        work_dir_root: str = "./.agent_workdir",
        session_id: Optional[str] = None,
    ):
        self.session_id = shortuuid.uuid() if session_id is None else session_id
        self.work_dir = os.path.abspath(os.path.join(work_dir_root, self.session_id))
        os.makedirs(self.work_dir, exist_ok=True)
        logger.info(f"LocalPythonExecutor work_dir: {self.work_dir}")

        self.km = None
        self.kc = None
        self._initialize_kernel()

        self.state = {}
        self.max_print_outputs_length = max_print_outputs_length or DEFAULT_MAX_LEN_OUTPUT

    def _initialize_kernel(self):
        """Starts the Jupyter kernel and executes initialization code."""
        logger.info("Initializing Jupyter kernel...")
        try:
            self.km, self.kc = start_new_kernel(kernel_name="python")
            logger.info("Kernel started.")
            init_code_formatted = INIT_CODE.format(work_dir=self.work_dir)
            logger.debug(f"Executing init code:\n{init_code_formatted}")
            # Use a synchronous helper to run the async execution within __init__
            init_results = self._execute_sync(init_code_formatted, timeout=TIMEOUT)
            logger.debug(f"Init code results: {init_results}")
            if init_results['status'] != "ok":
                stderr = extract_stderr(init_results)
                traceback = init_results.get('traceback', '')
                raise InterpreterError(f"Failed to initialize kernel environment. Error: {stderr}\n{traceback}")
            logger.info("Kernel initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing kernel: {e}", exc_info=True)
            self.cleanup() # Attempt cleanup on failure
            raise InterpreterError(f"Failed to start or initialize Jupyter kernel: {e}") from e

    def _execute_sync(self, code: str, timeout: int = TIMEOUT, silent: bool = False, store_history: bool = True, stop_on_error: bool = True) -> Dict[str, Any]:
        """Synchronous wrapper for executing code in the kernel (for use in __init__, __del__)."""
        if not self.kc:
            raise InterpreterError("Kernel client is not available.")

        msg_id = self.kc.execute(
            code=code, silent=silent, store_history=store_history, stop_on_error=stop_on_error
        )

        try:
            # Find an appropriate shell message (not kernel_info_reply)
            reply = None
            attempts = 0
            max_attempts = 5
            
            while attempts < max_attempts:
                reply = self.kc.get_shell_msg(timeout=timeout)
                
                # Skip kernel_info_reply messages
                if reply["header"]["msg_type"] == "kernel_info_reply":
                    logger.debug("Skipping kernel_info_reply message in sync mode")
                    attempts += 1
                    continue
                
                # Validate that the message is for this execution
                if reply["parent_header"].get("msg_id") != msg_id:
                    logger.warning(f"Message parent_header.msg_id '{reply['parent_header'].get('msg_id')}' != expected '{msg_id}'")
                    attempts += 1
                    continue
                    
                if reply["header"]["msg_type"] != "execute_reply":
                    logger.warning(f"Message header.msg_type '{reply['header']['msg_type']}' != expected 'execute_reply'")
                    attempts += 1
                    continue
                    
                # If we get here, we have a valid reply
                break
                
            if attempts >= max_attempts:
                logger.error("Could not get appropriate shell message after multiple attempts. Resetting kernel.")
                self._reset_kernel()
                raise InterpreterError("Did not receive execute_reply after multiple attempts. Kernel has been reset.")

        except Exception as e:
            logger.error(f"Error getting execute_reply: {e}. Resetting kernel.")
            self._reset_kernel()
            raise InterpreterError(f"The Jupyter kernel may have stalled or crashed. It has been forcibly restarted. Error: {e}") from e

        # IOPub handling for sync context
        output_msgs = []
        try:
            # First, check for busy status
            busy_msg = self.kc.iopub_channel.get_msg(timeout=1)
            if busy_msg["msg_type"] == "status" and busy_msg["content"]["execution_state"] == "busy" and busy_msg["parent_header"].get("msg_id") == msg_id:
                logger.debug("Got expected busy status message")
            else:
                logger.warning(f"First IOPub message was not the expected busy status: {busy_msg['msg_type']}")
            
            # Then collect output messages until we get an idle status
            while True:
                msg = self.kc.iopub_channel.get_msg(timeout=1)
                
                # Skip messages not related to this execution
                if msg["parent_header"].get("msg_id") != msg_id:
                    logger.debug(f"Skipping IOPub message with wrong msg_id: {msg['parent_header'].get('msg_id')}")
                    continue
                    
                if msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
                    break
                elif msg["msg_type"] == "execute_input":
                    # Skip execute_input messages
                    continue
                else:
                    output_msgs.append(msg)
                    
        except Exception as e:
            logger.warning(f"Exception while reading IOPub messages in sync mode: {e}. Output might be incomplete.")

        tb = "\n".join(reply["content"].get("traceback", []))
        if tb:
            tb = strip_ansi_codes(tb)

        results = {"status": reply["content"]["status"], "traceback": tb, "outputs": []}
        for msg in output_msgs:
            content = msg.get("content", {})
            msg_type = msg.get("msg_type")
            if msg_type == "stream":
                if content.get("name") == "stdout":
                    results["outputs"].append({"stdout": content.get("text", "")})
                elif content.get("name") == "stderr":
                    results["outputs"].append({"stderr": content.get("text", "")})
            elif msg_type == "error":
                results["outputs"].append({"error": f'{content.get("ename", "UnknownError")}: {content.get("evalue", "")}'})
            elif "data" in content:
                results["outputs"].append({msg_type: content.get("data", {})})
            else:
                logger.warning(f"Processing unknown message type: {msg_type} with content: {content}")

        # Basic file saving for display data (sync version)
        for output in results["outputs"]:
            data_payload = None
            if output.get("display_data"):
                data_payload = output.get("display_data")
            elif output.get("execute_result"):
                data_payload = output.get("execute_result")

            if data_payload:
                key = shortuuid.uuid()
                for type_name in list(data_payload.keys()):
                    data = data_payload[type_name]
                    file_path = None
                    try:
                        if type_name == 'image/png':
                            file_path = os.path.join(self.work_dir, f"{key}.png")
                            with open(file_path, 'wb') as f:
                                f.write(base64.b64decode(data))
                            del data_payload[type_name] # Remove encoded data after saving
                            data_payload['image/png'] = f"Image saved to: {os.path.basename(file_path)}" # Add reference
                        elif type_name in ('image/jpeg', 'image/jpg'):
                            file_path = os.path.join(self.work_dir, f"{key}.jpeg")
                            with open(file_path, 'wb') as f:
                                f.write(base64.b64decode(data))
                            del data_payload[type_name]
                            data_payload['image/jpeg'] = f"Image saved to: {os.path.basename(file_path)}"
                        elif type_name == 'text/plain' and len(data) > 1000:
                            file_path = os.path.join(self.work_dir, f"{key}.txt")
                            with open(file_path, 'w') as f:
                                f.write(data)
                            data_payload[type_name] = data[:1000] + f"... (Full content saved to: {os.path.basename(file_path)})"
                        # Add handling for other types if needed (e.g., application/json)

                    except Exception as e:
                        logger.error(f"Error saving display data type {type_name} to {file_path}: {e}")

        return results

    def _reset_kernel(self):
        """Shuts down the current kernel and starts a new one."""
        logger.warning("Resetting Jupyter kernel...")
        self.cleanup()
        self._initialize_kernel()

    async def __call__(self, code_action: str) -> Tuple[Any, str, bool]:
        """
        Execute Python code asynchronously using the Jupyter kernel.

        Args:
            code_action: The Python code to execute

        Returns:
            A tuple containing (output, logs, is_final_answer)
            - output: Represents the primary result (e.g., from display_data or execute_result) or None.
            - logs: Combined stdout and stderr.
            - is_final_answer: True if the code outputted FinalAnswerException: , otherwise False.
        """
        loop = asyncio.get_event_loop()
        if not self.kc:
            raise InterpreterError("Jupyter kernel is not initialized or has crashed.")

        exec_result = {}
        try:
            # Run the kernel execution logic asynchronously
            exec_result = await loop.run_in_executor(None, lambda: self._execute_sync(code_action))

            # Process results
            stdout = extract_stdout(exec_result)
            stderr = extract_stderr(exec_result)
            error = extract_error(exec_result) # Specific errors from error messages
            traceback = exec_result.get("traceback", "") # Traceback from execute_reply
            
            # Some outputs may contain escape sequences represented as string literals
            # Attempt to decode string literals for proper processing
            def clean_output(text):
                if not text:
                    return ""
                
                # Handle byte strings
                if isinstance(text, bytes):
                    try:
                        text = text.decode('utf-8')
                    except UnicodeDecodeError:
                        text = text.decode('latin1')  # Fallback
                
                # Handle escape sequences in string representations
                if isinstance(text, str) and ('\\x1b' in text or '\\033' in text):
                    try:
                        # Try to interpret escape sequences literally
                        text = text.encode().decode('unicode_escape')
                    except (UnicodeError, AttributeError):
                        # If that fails, just use the original text
                        pass
                
                # Apply ANSI code stripping
                return strip_ansi_codes(text)
            
            # Clean all outputs
            stdout = clean_output(stdout)
            stderr = clean_output(stderr)
            error = clean_output(error)
            traceback = clean_output(traceback)

            logs = ""
            if stdout:
                logs += f"stdout:\n{stdout}\n"
            if stderr:
                logs += f"stderr:\n{stderr}\n"

            # Combine errors and traceback
            error_report = ""
            if error:
                error_report += f"Kernel Error: {error}\n"
            if traceback:
                error_report += f"Traceback:\n{traceback}\n"

            # Check if final_answer was provided via marker in stdout
            is_final_answer = False
            final_answer_content = None
            
            # Look for FINAL_ANSWER_MARKER in stdout
            
            if FINAL_ANSWER_MARKER in error:
                is_final_answer = True
                # Extract everything after the marker
                final_answer_content = error.split(FINAL_ANSWER_MARKER, 1)[1].strip()
            if exec_result["status"] == "ok" or is_final_answer:
                output_data = final_answer_content if is_final_answer else self._extract_primary_output(exec_result["outputs"])
                # Strip ANSI codes from output data if it's a string
                if isinstance(output_data, str):
                    output_data = strip_ansi_codes(output_data)
                # Truncate logs if needed
                logs = truncate_content(logs, max_length=self.max_print_outputs_length)
                return output_data, logs, is_final_answer
            else:
                # Execution failed
                final_log = logs + error_report
                final_log = truncate_content(final_log, max_length=self.max_print_outputs_length)
                raise InterpreterError(f"Code execution failed.\n{final_log}")

        except Exception as e:
            # Catch potential timeouts, kernel crashes, or processing errors
            logger.error(f"Error during kernel execution or processing: {e}", exc_info=True)
            # Attempt to get logs even if processing failed
            stdout = extract_stdout(exec_result) if exec_result else ""
            stderr = extract_stderr(exec_result) if exec_result else ""
            
            # Clean outputs
            def clean_output(text):
                if not text:
                    return ""
                    
                # Handle byte strings
                if isinstance(text, bytes):
                    try:
                        text = text.decode('utf-8')
                    except UnicodeDecodeError:
                        text = text.decode('latin1')  # Fallback
                
                # Handle escape sequences in string representations
                if isinstance(text, str) and ('\\x1b' in text or '\\033' in text):
                    try:
                        # Try to interpret escape sequences literally
                        text = text.encode().decode('unicode_escape')
                    except (UnicodeError, AttributeError):
                        # If that fails, just use the original text
                        pass
                
                # Apply ANSI code stripping
                return strip_ansi_codes(text)
                
            stdout = clean_output(stdout)
            stderr = clean_output(stderr)
            
            logs = f"stdout:\n{stdout}\nstderr:\n{stderr}\n"
            logs = truncate_content(logs, max_length=self.max_print_outputs_length)

            if isinstance(e, InterpreterError):
                # Add existing logs to the interpreter error message if available
                e.args = (f"{e.args[0]}\nLogs:\n{logs}",)
                raise
            else:
                raise InterpreterError(f"Async execution error: {type(e).__name__}: {str(e)}\nLogs:\n{logs}") from e

    def _validate_message(self, msg, expected_type, expected_parent_id):
        """Validate that a message has the expected type and parent id."""
        if msg["parent_header"].get("msg_id") != expected_parent_id:
            logger.warning(f"Message parent_header.msg_id '{msg['parent_header'].get('msg_id')}' != expected '{expected_parent_id}'")
            return False
        
        if expected_type is not None and msg["header"]["msg_type"] != expected_type:
            logger.warning(f"Message header.msg_type '{msg['header']['msg_type']}' != expected '{expected_type}'")
            return False
            
        return True

    def _get_non_kernel_info_reply(self, msg_id, timeout=None):
        """Get a shell message that's not a kernel_info_reply."""
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            try:
                shell_msg = self.kc.get_shell_msg(timeout=timeout)
                
                if shell_msg["header"]["msg_type"] == "kernel_info_reply":
                    logger.debug("Skipping kernel_info_reply message")
                    continue
                    
                return shell_msg
            except Exception as e:
                logger.warning(f"Error getting shell message (attempt {attempts+1}/{max_attempts}): {e}")
                attempts += 1
                
        # If we've exhausted our attempts
        raise InterpreterError(f"Failed to get appropriate shell message after {max_attempts} attempts")

    def _execute_kernel_code(self, code: str, timeout: int = TIMEOUT, silent: bool = False, store_history: bool = True, stop_on_error: bool = True) -> Dict[str, Any]:
        """Internal async helper to execute code and process messages."""
        if not self.kc:
            raise InterpreterError("Kernel client is not available.")

        msg_id = self.kc.execute(
            code=code, silent=silent, store_history=store_history, stop_on_error=stop_on_error
        )

        reply = None
        output_msgs = []

        try:
            # Get execute_reply
            reply = self._get_non_kernel_info_reply(msg_id, timeout)
            
            # Validate the reply
            if not self._validate_message(reply, "execute_reply", msg_id):
                logger.warning("Received reply with unexpected message type or ID")
            
            # Get IOPub messages until idle
            while True:
                iopub_msg = self.kc.iopub_channel.get_msg(timeout=timeout)
                
                # Validate the message is for this execution
                if not self._validate_message(iopub_msg, None, msg_id):
                    continue

                if iopub_msg["msg_type"] == "status":
                    if iopub_msg["content"]["execution_state"] == "idle":
                        break # Execution finished
                elif iopub_msg["msg_type"] == "execute_input":
                    # Optional: log or verify the input code
                    pass
                else:
                    # Collect relevant output messages
                    output_msgs.append(iopub_msg)

        except asyncio.TimeoutError:
            logger.error(f"Timeout ({timeout}s) waiting for kernel response. Resetting kernel.")
            self._reset_kernel() # Attempt reset
            raise InterpreterError(f"Timeout waiting for Jupyter kernel response after {timeout} seconds. Kernel has been reset.") from None
        except Exception as e:
            logger.error(f"Error communicating with kernel: {e}. Resetting kernel.")
            self._reset_kernel() # Attempt reset
            raise InterpreterError(f"Error communicating with Jupyter kernel: {e}. Kernel has been reset.") from e

        # Process results after successful communication
        tb = "\n".join(reply["content"].get("traceback", []))
        if tb:
            tb = strip_ansi_codes(tb)

        results = {"status": reply["content"]["status"], "traceback": tb, "outputs": []}
        for msg in output_msgs:
            content = msg.get("content", {})
            msg_type = msg.get("msg_type")
            if msg_type == "stream":
                if content.get("name") == "stdout":
                    results["outputs"].append({"stdout": content.get("text", "")})
                elif content.get("name") == "stderr":
                    results["outputs"].append({"stderr": content.get("text", "")})
            elif msg_type == "error":
                results["outputs"].append({"error": f'{content.get("ename", "UnknownError")}: {content.get("evalue", "")}'})
            elif "data" in content:
                # Process display data (potentially saving files)
                self._process_display_data(results["outputs"], msg_type, content.get("data", {}))
            else:
                logger.warning(f"Processing unknown message type: {msg_type} with content: {content}")

        return results
        
    def _process_display_data(self, output_list: list, msg_type: str, data_payload: dict):
        """Processes display_data/execute_result, saving files asynchronously."""
        processed_payload = {}
        key = shortuuid.uuid() # Unique key for potential file saving

        for type_name, data in data_payload.items():
            file_path = None
            try:
                if type_name == 'image/png':
                    file_path = os.path.join(self.work_dir, f"{key}.png")
                    image_bytes = base64.b64decode(data)
                    with open(file_path, 'wb') as file:
                        file.write(image_bytes)
                    processed_payload[type_name] = f"Image saved to: {os.path.basename(file_path)}"
                elif type_name in ('image/jpeg', 'image/jpg'):
                    file_path = os.path.join(self.work_dir, f"{key}.jpeg")
                    image_bytes = base64.b64decode(data)
                    with open(file_path, 'wb') as file:
                        file.write(image_bytes)
                    processed_payload[type_name] = f"Image saved to: {os.path.basename(file_path)}"
                elif type_name == 'text/plain' and isinstance(data, str) and len(data) > 1000:
                    file_path = os.path.join(self.work_dir, f"{key}.txt")
                    with open(file_path, 'w') as file:
                        file.write(data)
                    processed_payload[type_name] = data[:1000] + f"... (Full content saved to: {os.path.basename(file_path)})"
                elif type_name.startswith('application/'): # Example: save JSON
                    file_path = os.path.join(self.work_dir, f"{key}.{type_name.split('/')[-1]}")
                    try:
                        # Attempt to dump nicely if it's JSON-like
                        json_data = json.dumps(data, indent=2)
                        with open(file_path, 'w') as file:
                            file.write(json_data)
                    except (TypeError, ValueError):
                        # Fallback to string representation if not JSON serializable
                        with open(file_path, 'w') as file:
                            file.write(str(data))
                    processed_payload[type_name] = f"Data saved to: {os.path.basename(file_path)}"
                else:
                    # Keep other data types as is
                    processed_payload[type_name] = data
            except Exception as e:
                logger.error(f"Error saving display data type {type_name} to {file_path}: {e}")
                processed_payload[type_name] = f"[Error saving data: {e}]" # Keep original key, add error note

        output_list.append({msg_type: processed_payload})


    def _extract_primary_output(self, outputs: List[Dict]) -> Any:
        """Extracts the most relevant output, preferring display_data/execute_result."""
        primary_output = None
        for item in reversed(outputs): # Check latest outputs first
            if "display_data" in item:
                primary_output = item["display_data"]
                break
            if "execute_result" in item:
                primary_output = item["execute_result"]
                break
        return primary_output


    async def send_variables(self, variables: dict):
        """
        Send variables to the kernel using cloudpickle for robust serialization.
        
        Args:
            variables: Dictionary of variables to send to the kernel
        """
        if not self.kc:
            logger.warning("Cannot send variables, kernel not available.")
            return
            
        loop = asyncio.get_event_loop()
        
        variables = variables or {}
            
        # First, ensure cloudpickle is available in the kernel
        setup_code = '''
# Ensure cloudpickle is available
try:
    import cloudpickle
except ImportError:
    import sys
    import subprocess
    print("Installing cloudpickle...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cloudpickle"])
    import cloudpickle

import base64
import traceback

# Dictionary to track successfully loaded variables
_loaded_vars = {}
'''
        # Execute setup code
        try:
            setup_result = await loop.run_in_executor(None, lambda: self._execute_sync(setup_code))
            if setup_result["status"] != "ok":
                logger.error(f"Failed to set up cloudpickle in kernel: {setup_result}")
                return
        except Exception as e:
            logger.error(f"Error setting up cloudpickle: {e}", exc_info=True)
            return
            
        # Import cloudpickle in this process too
        try:
            import cloudpickle
        except ImportError:
            import subprocess
            import sys
            logger.info("Installing cloudpickle in main process...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cloudpickle"])
            import cloudpickle
            
        # Process each variable
        for key, value in variables.items():
            if not isinstance(key, str):
                logger.warning(f"Variable key must be a string, got {type(key)}. Skipping.")
                continue
                
            # Attempt to pickle the variable
            try:
                # Serialize with cloudpickle
                pickled_data = cloudpickle.dumps(value)
                # Use base64 to avoid any string escaping issues
                b64_data = base64.b64encode(pickled_data).decode('ascii')
                
                # Create code to unpickle in the kernel
                var_code = f'''
try:
    import base64
    import cloudpickle
    
    # Decode base64 and unpickle
    pickled_data = base64.b64decode('{b64_data}')
    {key} = cloudpickle.loads(pickled_data)
    
    # Track successful loading
    _loaded_vars["{key}"] = True
    
    print(f"Variable '{key}' of type {{type({key}).__name__}} loaded successfully")
except Exception as e:
    print(f"Error loading variable '{key}': {{e}}")
    traceback.print_exc()
    _loaded_vars["{key}"] = False
'''
                
                # Execute the variable loading code
                result = await loop.run_in_executor(None, lambda: self._execute_sync(var_code))
                
                if result["status"] != "ok":
                    logger.warning(f"Failed to load variable '{key}' in kernel: {extract_stderr(result)}")
                else:
                    stdout = extract_stdout(result)
                    if f"Variable '{key}'" in stdout and "loaded successfully" in stdout:
                        logger.debug(f"Successfully sent variable '{key}' to kernel: {stdout}")
                    else:
                        logger.warning(f"Issue sending variable '{key}' to kernel: {stdout}")
                        
            except Exception as e:
                logger.error(f"Error serializing variable '{key}': {e}", exc_info=True)
                
        # Check which variables were loaded successfully
        check_code = "print(f\"Loaded variables: {list(_loaded_vars.keys())}\")"
        try:
            check_result = await loop.run_in_executor(None, lambda: self._execute_sync(check_code))
            if check_result["status"] == "ok":
                logger.info(extract_stdout(check_result))
        except Exception as e:
            logger.error(f"Error checking loaded variables: {e}")

    async def send_tools(self, tools: Dict[str, Tool]):
        """
        Send tool functions to the kernel using cloudpickle for robust serialization.
        
        Args:
            tools: Dictionary mapping tool names to Tool objects
        """
        loop = asyncio.get_event_loop()
        if not self.kc or not tools:
            return
            
        logger.info(f"Sending {len(tools)} tools to kernel")
        
        # First, ensure cloudpickle is available in the kernel
        setup_code = '''
# Ensure cloudpickle is available
try:
    import cloudpickle
except ImportError:
    import sys
    import subprocess
    print("Installing cloudpickle...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cloudpickle"])
    import cloudpickle

import base64
import traceback
import inspect
import functools
import asyncio

# Create a dict to store tools
_tools = {}

# Helper to make async functions work synchronously in the Jupyter kernel
def make_sync(async_func):
    @functools.wraps(async_func)
    def sync_wrapper(*args, **kwargs):
        # Get or create an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async function to completion
        return loop.run_until_complete(async_func(*args, **kwargs))
    
    return sync_wrapper

# Helper to install a function into the global namespace with a given name
def install_function(name, func):
    globals()[name] = func
    return func
'''

        # Execute setup code
        try:
            setup_result = await loop.run_in_executor(None, lambda: self._execute_sync(setup_code))
            if setup_result["status"] != "ok":
                logger.error(f"Failed to set up cloudpickle for tools: {setup_result}")
                return
        except Exception as e:
            logger.error(f"Error setting up cloudpickle for tools: {e}", exc_info=True)
            return
            
        # Import cloudpickle in this process too
        try:
            import cloudpickle
            import inspect
        except ImportError:
            import subprocess
            import sys
            logger.info("Installing cloudpickle in main process...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cloudpickle"])
            import cloudpickle
            import inspect
        
        # Process each tool
        for tool_name, tool in tools.items():
            try:
                # Get the function from the tool
                if hasattr(tool, 'function'):
                    func = tool.function
                else:
                    func = tool  # Assume it's the function directly
                
                # Skip non-callable tools
                if not callable(func):
                    logger.warning(f"Tool {tool_name} is not callable, skipping")
                    continue
                
                # Check if it's an async function
                is_async = inspect.iscoroutinefunction(func)
                
                # Serialize with cloudpickle
                pickled_func = cloudpickle.dumps(func)
                # Use base64 to avoid string escaping issues
                b64_func = base64.b64encode(pickled_func).decode('ascii')
                
                # Code to deserialize and install the function
                func_code = f'''
try:
    import base64
    import cloudpickle
    import inspect
    
    # Decode and unpickle the function
    pickled_func = base64.b64decode('{b64_func}')
    func = cloudpickle.loads(pickled_func)
    
    # Store in tools dictionary
    _tools["{tool_name}"] = func
    
    # Check if it's async and make a sync version too
    is_async = inspect.iscoroutinefunction(func)
    
    # Install in global namespace
    install_function("{tool_name}", func)
    
    # If async, also create a sync version
    if is_async:
        sync_func = make_sync(func)
        install_function("{tool_name}_sync", sync_func)
        print(f"Tool '{tool_name}' (async) and '{tool_name}_sync' loaded successfully")
    else:
        print(f"Tool '{tool_name}' loaded successfully")
        
except Exception as e:
    print(f"Error loading tool '{tool_name}': {{e}}")
    traceback.print_exc()
'''
                
                # Execute the function loading code
                result = await loop.run_in_executor(None, lambda: self._execute_sync(func_code))
                
                if result["status"] != "ok":
                    logger.error(f"Failed to load tool {tool_name}: {extract_stderr(result) or result.get('traceback', '')}")
                else:
                    stdout = extract_stdout(result)
                    if f"Tool '{tool_name}'" in stdout and "loaded successfully" in stdout:
                        logger.info(f"Tool {tool_name} loaded into kernel: {stdout}")
                    else:
                        logger.warning(f"Issue loading tool {tool_name}: {stdout}")
                        
            except Exception as e:
                logger.error(f"Error serializing tool {tool_name}: {e}", exc_info=True)
                
        # Check which tools were loaded
        check_code = "print(f\"Loaded tools: {list(_tools.keys())}\")"
        try:
            check_result = await loop.run_in_executor(None, lambda: self._execute_sync(check_code))
            if check_result["status"] == "ok":
                logger.info(extract_stdout(check_result))
        except Exception as e:
            logger.error(f"Error checking loaded tools: {e}")

    def cleanup(self):
        """Shuts down the kernel manager and client."""
        logger.info("Cleaning up Jupyter kernel...")
        if hasattr(self, 'kc') and self.kc:
            try:
                self.kc.stop_channels()
                logger.info("Kernel channels stopped.")
            except Exception as e:
                logger.warning(f"Error stopping kernel channels: {e}")
            self.kc = None

        if hasattr(self, 'km') and self.km:
            try:
                if self.km.is_alive():
                    self.km.shutdown_kernel(now=True)
                    logger.info("Kernel shutdown requested.")
                else:
                    logger.info("Kernel already shutdown.")
            except Exception as e:
                logger.error(f"Error during kernel shutdown: {e}")
            self.km = None
        logger.info("Kernel cleanup finished.")

        
    def __del__(self):
        """Clean up the executor when the object is garbage collected"""
        self.cleanup()


__all__ = ["LocalPythonExecutor", "InterpreterError"]
