import pytest
import os
from schema_agents.tools.code_interpreter import CodeInterpreter, extract_stdout, extract_stderr

class TestCodeInterpreter:
    @pytest.fixture(autouse=True)
    def init_code_interpreter(self):
        self.work_dir_root = "./.data"
        os.makedirs(self.work_dir_root, exist_ok=True)
        self.code_interpreter = CodeInterpreter(work_dir_root=self.work_dir_root)

    def test_execute_code(self):
        result = self.code_interpreter.execute_code("print('hello world')")
        stdout = extract_stdout(result)
        assert result["status"] == "ok", f"expected 'ok', got {result['status']}"
        assert stdout == "hello world\n", f"expected 'hello world\\n', got {stdout}"

        result = self.code_interpreter.execute_code("import numpy as np; print(np.random.rand(3))")
        assert result["status"] == "ok", f"expected 'ok', got {result['status']}"

        result = self.code_interpreter.execute_code("x = 1/0") # to induce ZeroDivisionError
        assert result["status"] == "error", f"expected 'error', got {result['status']}"
        assert "ZeroDivisionError" in result["traceback"], f"expected 'ZeroDivisionError' in traceback"

    def test_execute_query(self):
        output = self.code_interpreter.execute_query("say 'hello world'")
        assert "hello world" in output.lower(), f"expected 'hello world' in output, got {output}"

        output = self.code_interpreter.execute_query("create a random array and make sure the shape is (3,3)")
        assert "(3, 3)" in output, f"expected '(3, 3)' in output, got {output}"

        output = self.code_interpreter.execute_query("x = 1/0")  # to induce division by zero
        # The model will use try and except to capture the error, and print the error message.
        assert "division by zero" in output, f"expected 'division by zero' in output, got {output}"
        
        output = self.code_interpreter.execute_query("make a simple plot")
        assert "successfully executed" in output, f"expected 'successfully executed' in output, got {output}"
        output = self.code_interpreter.execute_query("save the plot into a png file named 'plot.png'")
        assert "successfully executed" in output, f"expected 'successfully executed' in output, got {output}"
        assert os.path.exists(os.path.join(self.code_interpreter.work_dir, "plot.png")), f"expected 'plot.png' in {self.code_interpreter.work_dir}, got {os.listdir(self.code_interpreter.work_dir)}"
        

    def test_tear_down(self):
        self.code_interpreter.tearDown()
        assert self.code_interpreter.km is None, "Kernel Manager is not shut down correctly."
        assert self.code_interpreter.kc is None, "Kernel Client is not shut down correctly."
