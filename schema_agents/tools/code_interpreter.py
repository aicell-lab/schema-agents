import os
import re
import inspect
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
from jupyter_client.manager import start_new_kernel
from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel
from jupyter_client.utils import run_sync
from .msgspec_v5 import validate_message
import os
import re
import shortuuid
import base64
import json
import logging

# Set logging level to INFO for more detailed output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def strip_ansi_codes(s):
    return re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', s)

TIMEOUT=15

INIT_CODE = """%matplotlib inline
%cd {work_dir}
import sys
sys.path.append('{work_dir}')
"""

KERNEL_INFO_CODE="""
import platform
import pkg_resources
python_version = platform.python_version()
# Get list of installed packages and their versions
installed_packages = [f'{d.project_name}-{d.version}' for d in pkg_resources.working_set]
print(f"PYTHON VERSION: {python_version}, INSTALLED PACKAGES: {','.join(installed_packages)}")
"""

SYSTEM_PROMPT = """
Act as a professional python programmer, you help user to generate useful python code.
When you send a message containing Python code (quoted with ```python) to the user, it will be executed in a stateful Jupyter notebook environment. Jupyter will respond with the output of the execution or time out after 120.0 
seconds. The drive at '{work_dir}' can be used to save and persist user files. Internet access for this session is disabled. Do not make external web requests or API calls as they will fail.
Input should be valid python code for this particular jupyter notebook kernel. Here is more details about the jupyter environment:
{kernel_info}

You MUST ensure that the code generated is compatible with the Jupyter Notebook environment.
"""

FORMAT_INSTRUCTION = """If you want to see the output of a value, you should print it out with `print(...)`.
Example:
User: print hello world
```
print("hello world")
```
In the following message, the user might provide feedback on the result of the code execution via `Jupyter: ....`. The feedback will be used to improve the model.
"""

ERROR_FIX_PROMPT = "Error while executing the code above: {error} \nPlease fix the error and try again. If needed, you can use `import subprocess;subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'new_package'])` to install a new package; DO NOT generate the same code or error twice."

def ensure_sync(func):
    if inspect.iscoroutinefunction(func):
        return run_sync(func)
    return func

def extract_code_blocks(text):
    pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text, re.MULTILINE)
    return matches

def extract_stdout(results):
    output = results["outputs"]
    return "\n".join([item['stdout'] for item in output if 'stdout' in item])

def extract_stderr(results):
    output = results["outputs"]
    return "\n".join([item['stderr'] for item in output if 'stderr' in item])

def extract_error(results):
    output = results["outputs"]
    return "\n".join([item['error'] for item in output if 'error' in item])

def extract_execute_result(results):
    output = results["outputs"]
    return "\n".join([item['execute_result'] for item in output if 'execute_result' in item])

def extract_display_data(results):
    output = results["outputs"]
    return "\n".join([item['display_data'] for item in output if 'display_data' in item])

def get_available_kernels():
    """Get a list of available Jupyter kernels."""
    try:
        kernel_spec_manager = KernelSpecManager()
        kernel_specs = kernel_spec_manager.find_kernel_specs()
        logger.info(f"Found kernel specs: {kernel_specs}")
        return list(kernel_specs.keys())
    except Exception as e:
        logger.error(f"Error getting available kernels: {str(e)}")
        return []

def find_python_kernel():
    """Find a suitable Python kernel from available kernels."""
    available_kernels = get_available_kernels()
    logger.info(f"Available kernels in find_python_kernel: {available_kernels}")
    
    if not available_kernels:
        logger.warning("No available kernels found")
        return None
    
    # Define priority order for kernel selection
    priority_kernels = [
        "python3",  # Standard Python 3 kernel
        "python",   # Generic Python kernel
        "jupyterlab", # JupyterLab kernel
        "ipykernel", # IPython kernel
    ]
    
    # First try to find kernels from our priority list
    for kernel in priority_kernels:
        if kernel in available_kernels:
            logger.info(f"Found priority kernel: {kernel}")
            return kernel
    
    # If none of the priority kernels are available, look for any kernel with 'python' in the name
    python_kernels = [k for k in available_kernels if 'python' in k.lower()]
    if python_kernels:
        logger.info(f"Found Python kernel: {python_kernels[0]}")
        return python_kernels[0]
    
    # If no Python kernels found, return the first available kernel
    if available_kernels:
        logger.info(f"Using first available kernel: {available_kernels[0]}")
        return available_kernels[0]
    
    logger.warning("No kernels found after all attempts")
    return None

class CodeInterpreter:
    MAIN_SESSION_ID = "main"

    def __init__(self, session_id=None, work_dir_root="./.code-interpreter", kernel_name=None):
        self.store = {}
        self._last_output = ""
        self.session_id = shortuuid.uuid() if session_id is None else session_id
        self.work_dir = os.path.join(work_dir_root, self.session_id)
        self.kernel_name = kernel_name
        logger.info(f"CodeInterpreter initialized with kernel_name: {kernel_name}")
        assert os.path.exists(work_dir_root), f"work_dir_root {work_dir_root} does not exist"
        os.makedirs(self.work_dir, exist_ok=True)
        self.initialize()

    def initialize(self):
        # If no kernel name is specified, try to find a suitable one
        if not self.kernel_name:
            logger.info("No kernel name specified, finding a suitable one")
            self.kernel_name = find_python_kernel()
            if not self.kernel_name:
                logger.error("No suitable Jupyter kernel found")
                raise RuntimeError("No suitable Jupyter kernel found. Please install a Python kernel or specify a kernel name.")
        
        try:
            logger.info(f"Starting kernel: {self.kernel_name}")
            self.km, self.kc = start_new_kernel(kernel_name=self.kernel_name)
            logger.info("Kernel started successfully")
        except NoSuchKernel as e:
            logger.error(f"Kernel not found: {self.kernel_name}, error: {str(e)}")
            available_kernels = get_available_kernels()
            if available_kernels:
                # If the specified kernel is not available, try to use an available one
                self.kernel_name = find_python_kernel()
                logger.warning(f"Specified kernel not found. Using {self.kernel_name} instead.")
                self.km, self.kc = start_new_kernel(kernel_name=self.kernel_name)
            else:
                logger.error("No Jupyter kernels available")
                raise RuntimeError(f"No Jupyter kernels available. Please install at least one kernel.")
        except Exception as e:
            logger.error(f"Error starting kernel: {str(e)}")
            raise
        
        results = self.execute_code(INIT_CODE.format(work_dir=self.work_dir) + KERNEL_INFO_CODE)
        assert results['status'] == "ok", f"Failed to get kernel info, output: {results}"
        kernel_info = extract_stdout(results).replace('\n', ' ')
        self.system_prompt = SYSTEM_PROMPT.format(work_dir=self.work_dir, kernel_info=kernel_info)

    def reset(self):
        self.tearDown()
        self.initialize()

    def tearDown(self):
        # Check if the kernel client and kernel manager exist
        if self.kc:
            # Send a shutdown request to the kernel
            self.kc.shutdown()
            self.kc = None

        if self.km:
            # Shut down the kernel
            self.km.shutdown_kernel()
            self.km = None

    def assertEqual(self, a, b):
        assert a == b, f"{a} != {b}"

    def get_non_kernel_info_reply(self, timeout=None):
        while True:
            reply = self.kc.get_shell_msg(timeout=timeout)
            if reply["header"]["msg_type"] != "kernel_info_reply":
                return reply

    def get_data_from_message(self, output_msgs, msg_type):
        data_list = []
        for msg in output_msgs:
            if msg["msg_type"] == msg_type:
                data_list.append(msg["content"]["data"])
            else:
                continue
        if data_list:
            return data_list
        
    def execute_code(self, code, timeout=TIMEOUT, silent=False, store_history=True, stop_on_error=True):
        """Execute python code in a jupyter kernel and return the outcome.
        Arguments:
         - code (str): The code to execute, need to be a valid python code.
        Returns: a dict containing the STDOUT, STDERR, and the execution status.
        """
        msg_id = self.kc.execute(
            code=code, silent=silent, store_history=store_history, stop_on_error=stop_on_error
        )

        try:
            reply = self.get_non_kernel_info_reply(timeout=timeout)
            validate_message(reply, "execute_reply", msg_id)
        except Exception:
            self.reset()
            raise RuntimeError(f"The Jupyter kernel may have stalled or crashed, failing to provide an execute_reply within {timeout} seconds. It has been forcibly restarted! Please avoid using functions that interact with desktop UI, such as cv2.imshow, which could potentially crash the kernel.")

        busy_msg = ensure_sync(self.kc.iopub_channel.get_msg)(timeout=1)
        validate_message(busy_msg, "status", msg_id)
        self.assertEqual(busy_msg["content"]["execution_state"], "busy")

        output_msgs = []
        while True:
            msg = ensure_sync(self.kc.iopub_channel.get_msg)(timeout=0.1)
            validate_message(msg, msg["msg_type"], msg_id)
            if msg["msg_type"] == "status":
                self.assertEqual(msg["content"]["execution_state"], "idle")
                break
            elif msg["msg_type"] == "execute_input":
                self.assertEqual(msg["content"]["code"], code)
                continue
            output_msgs.append(msg)
        # Process output messages
        tb = "\n".join(reply["content"].get("traceback", []))
        if tb:
            tb = strip_ansi_codes(tb)
        results = {"status": reply["content"]["status"], "traceback": tb, "outputs": []}
        for msg in output_msgs:
            if msg["msg_type"] == "stream":
                if msg["content"]["name"] == "stdout":
                    results["outputs"].append({"stdout": msg["content"]["text"]})
                elif msg["content"]["name"] == "stderr":
                    results["outputs"].append({"stderr": msg["content"]["text"]})
            elif msg["msg_type"] == "error":
                results["outputs"].append({"error": f'{msg["content"]["ename"]}: {msg["content"]["evalue"]}'})
            elif "data" in msg["content"]:
                results["outputs"].append({msg["msg_type"]: msg["content"]["data"]})
            else:
                logger.warning("Unknown message type: %s", msg["msg_type"])

        # Here we process the execution results
        for output in results["outputs"]:
            if output.get("display_data"):
                display_data = output.get("display_data")
                display_type = "display_data"
            elif output.get("execute_result"):
                display_data = output.get("execute_result")
                display_type = "execute_result"
            else:
                continue
            key = shortuuid.uuid() # Define a unique key for the display_data
            for type_name in list(display_data.keys()):
                data = display_data[type_name]
                if type_name == 'text/plain':
                    if len(data) > 1000:
                        # truncate the text if it is too long
                        file_name = f"{key}.txt"
                        with open(os.path.join(self.work_dir, file_name), 'w') as file:
                            file.write(data)
                        display_data[type_name] = data[:1000] + "..."

                elif type_name == 'image/png':
                    file_extension = 'png'
                    image_bytes = base64.b64decode(data)
                    file_name = f"{key}.{file_extension}"
                    with open(os.path.join(self.work_dir, file_name), 'wb') as file:
                        file.write(image_bytes)
                    del display_data[type_name]
                elif type_name == 'image/jpeg' or type_name == 'image/jpg':
                    file_extension = 'jpeg'
                    image_bytes = base64.b64decode(data)
                    file_name = f"{key}.{file_extension}"
                    with open(os.path.join(self.work_dir, file_name), 'wb') as file:
                        file.write(image_bytes)
                    del display_data[type_name]
                else:
                    # for other types of display_data, save them into a JSON file
                    file_extension = 'json'
                    file_name = f"{key}.{file_extension}"
                    with open(os.path.join(self.work_dir, file_name), 'w') as file:
                        json.dump({'type': type_name, 'data': data}, file)

        return results
        
    def get_store(self, key):
        return self.store[key]

    def _create_summary(self, result):
        """
        Create a text summary to describe the execution result.
        """
        return json.dumps({k: result[k] for k in result.keys() if result[k]}, indent=1)

if __name__ == "__main__":
    work_dir_root = "./.data"
    os.makedirs(work_dir_root, exist_ok=True)
    ex = CodeInterpreter(work_dir_root=work_dir_root)
    try:
        # result = ex.execute_code("print('hello world')")
        # print("result for print hello world:", result)
        # result = ex.execute_code("import numpy as np; print(np.random.rand(3))")
        # print("result for numpy array", result)
        # result = ex.execute_code("import matplotlib.pyplot as plt; plt.plot([1,2,3,4]); plt.show()")
        # print("result for plotting", result)
        result = ex.execute_query("make a simple plot")
        print("simple plot:", result)
        result = ex.execute_query("save the plot into a png file")
        print("save plot:", result)
    except Exception as e:
        raise e
    finally:
        ex.tearDown()