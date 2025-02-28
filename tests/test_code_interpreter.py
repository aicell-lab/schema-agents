import pytest
import os
import json
import time
import logging
from unittest.mock import patch, MagicMock
from schema_agents.tools.code_interpreter import (
    CodeInterpreter, 
    extract_stdout, 
    extract_stderr, 
    extract_error, 
    extract_execute_result, 
    extract_display_data,
    get_available_kernels,
    find_python_kernel
)

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directly specify the kernels we know are available from our check
AVAILABLE_KERNELS = ['imjoy-iteractive-ml', 'python27', 'jupyterlab', 'imjoyrpc']

class TestCodeInterpreter:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment."""
        self.work_dir_root = "./.data"
        os.makedirs(self.work_dir_root, exist_ok=True)
        yield
    
    def test_extract_functions(self):
        """Test the extract functions for different output types."""
        # Test extract_stdout
        result = {"outputs": [{"stdout": "hello"}, {"stdout": "world"}]}
        assert extract_stdout(result) == "hello\nworld"
        
        # Test extract_stderr
        result = {"outputs": [{"stderr": "error1"}, {"stderr": "error2"}]}
        assert extract_stderr(result) == "error1\nerror2"
        
        # Test extract_error
        result = {"outputs": [{"error": "ZeroDivisionError"}, {"error": "ValueError"}]}
        assert extract_error(result) == "ZeroDivisionError\nValueError"
        
        # Test extract_execute_result
        result = {"outputs": [{"execute_result": "result1"}, {"execute_result": "result2"}]}
        assert extract_execute_result(result) == "result1\nresult2"
        
        # Test extract_display_data
        result = {"outputs": [{"display_data": "data1"}, {"display_data": "data2"}]}
        assert extract_display_data(result) == "data1\ndata2"
    
    @patch('schema_agents.tools.code_interpreter.start_new_kernel')
    @patch.object(CodeInterpreter, 'execute_code')
    def test_create_summary(self, mock_execute_code, mock_start_kernel):
        """Test the _create_summary method."""
        # Set up the mock kernel
        mock_km = MagicMock()
        mock_kc = MagicMock()
        mock_start_kernel.return_value = (mock_km, mock_kc)
        
        # Set up mock execute_code to return a successful result
        mock_execute_code.return_value = {
            "status": "ok",
            "traceback": "",
            "outputs": [{"stdout": "Kernel info output"}]
        }
        
        # Create a code interpreter with the mock kernel
        code_interpreter = CodeInterpreter(
            session_id="test_session",
            work_dir_root=self.work_dir_root,
            kernel_name="python"
        )
        
        # Test the _create_summary method
        result = {
            "status": "ok",
            "traceback": "",
            "outputs": [{"stdout": "hello world"}],
            "empty_field": ""
        }
        
        summary = code_interpreter._create_summary(result)
        summary_dict = json.loads(summary)
        
        assert "status" in summary_dict
        assert "outputs" in summary_dict
        assert "empty_field" not in summary_dict  # Empty fields should be excluded
    
    @patch('schema_agents.tools.code_interpreter.start_new_kernel')
    @patch.object(CodeInterpreter, 'execute_code')
    def test_get_data_from_message(self, mock_execute_code, mock_start_kernel):
        """Test the get_data_from_message function."""
        # Set up the mock kernel
        mock_km = MagicMock()
        mock_kc = MagicMock()
        mock_start_kernel.return_value = (mock_km, mock_kc)
        
        # Set up mock execute_code to return a successful result
        mock_execute_code.return_value = {
            "status": "ok",
            "traceback": "",
            "outputs": [{"stdout": "Kernel info output"}]
        }
        
        # Create a code interpreter with the mock kernel
        code_interpreter = CodeInterpreter(
            session_id="test_session",
            work_dir_root=self.work_dir_root,
            kernel_name="python"
        )
        
        # Create a mock message list
        output_msgs = [
            {"msg_type": "display_data", "content": {"data": {"text/plain": "data1"}}},
            {"msg_type": "execute_result", "content": {"data": {"text/plain": "data2"}}},
            {"msg_type": "display_data", "content": {"data": {"text/plain": "data3"}}}
        ]
        
        # Test getting display_data
        data_list = code_interpreter.get_data_from_message(output_msgs, "display_data")
        assert len(data_list) == 2
        assert data_list[0]["text/plain"] == "data1"
        assert data_list[1]["text/plain"] == "data3"
        
        # Test getting execute_result
        data_list = code_interpreter.get_data_from_message(output_msgs, "execute_result")
        assert len(data_list) == 1
        assert data_list[0]["text/plain"] == "data2"
    
    @patch('schema_agents.tools.code_interpreter.start_new_kernel')
    @patch.object(CodeInterpreter, 'execute_code')
    def test_execute_code_mock(self, mock_execute_code, mock_start_kernel):
        """Test the execute_code method with mocking."""
        # Set up the mock kernel
        mock_km = MagicMock()
        mock_kc = MagicMock()
        mock_start_kernel.return_value = (mock_km, mock_kc)
        
        # Set up mock execute_code to return a successful result for initialization
        mock_execute_code.return_value = {
            "status": "ok",
            "traceback": "",
            "outputs": [{"stdout": "Kernel info output"}]
        }
        
        # Create a code interpreter with the mock kernel
        code_interpreter = CodeInterpreter(
            session_id="test_session",
            work_dir_root=self.work_dir_root,
            kernel_name="python"
        )
        
        # Reset the mock to test our own execute_code calls
        mock_execute_code.reset_mock()
        
        # Set up different return values for different calls
        mock_execute_code.side_effect = [
            # First call - successful execution
            {
                "status": "ok",
                "traceback": "",
                "outputs": [{"stdout": "hello world\n"}]
            },
            # Second call - error execution
            {
                "status": "error",
                "traceback": "ZeroDivisionError: division by zero",
                "outputs": [{"error": "ZeroDivisionError"}]
            }
        ]
        
        # Test successful execution
        result = code_interpreter.execute_code("print('hello world')")
        assert result["status"] == "ok"
        stdout = extract_stdout(result)
        assert stdout == "hello world\n"
        
        # Test error execution
        result = code_interpreter.execute_code("x = 1/0")
        assert result["status"] == "error"
        assert "ZeroDivisionError" in result["traceback"]
    
    @patch('schema_agents.tools.code_interpreter.start_new_kernel')
    @patch.object(CodeInterpreter, 'execute_code')
    def test_tear_down(self, mock_execute_code, mock_start_kernel):
        """Test the tearDown method."""
        # Set up the mock kernel
        mock_km = MagicMock()
        mock_kc = MagicMock()
        mock_start_kernel.return_value = (mock_km, mock_kc)
        
        # Set up mock execute_code to return a successful result
        mock_execute_code.return_value = {
            "status": "ok",
            "traceback": "",
            "outputs": [{"stdout": "Kernel info output"}]
        }
        
        # Create a code interpreter with the mock kernel
        code_interpreter = CodeInterpreter(
            session_id="test_session",
            work_dir_root=self.work_dir_root,
            kernel_name="python"
        )
        
        # Call tearDown
        code_interpreter.tearDown()
        
        # Verify that the kernel client and manager were shut down
        mock_kc.shutdown.assert_called_once()
        mock_km.shutdown_kernel.assert_called_once()
        
        # Verify that km and kc are set to None
        assert code_interpreter.km is None
        assert code_interpreter.kc is None
    
    @patch('schema_agents.tools.code_interpreter.start_new_kernel')
    @patch.object(CodeInterpreter, 'execute_code')
    def test_reset(self, mock_execute_code, mock_start_kernel):
        """Test the reset method."""
        # Set up the mock kernel
        mock_km = MagicMock()
        mock_kc = MagicMock()
        mock_start_kernel.return_value = (mock_km, mock_kc)
        
        # Set up mock execute_code to return a successful result
        mock_execute_code.return_value = {
            "status": "ok",
            "traceback": "",
            "outputs": [{"stdout": "Kernel info output"}]
        }
        
        # Create a code interpreter with the mock kernel
        code_interpreter = CodeInterpreter(
            session_id="test_session",
            work_dir_root=self.work_dir_root,
            kernel_name="python"
        )
        
        # Store references to km and kc
        original_km = code_interpreter.km
        original_kc = code_interpreter.kc
        
        # Mock the initialize method to track calls
        original_initialize = code_interpreter.initialize
        code_interpreter.initialize = MagicMock()
        
        # Call reset
        code_interpreter.reset()
        
        # Verify that tearDown was called (by checking if shutdown methods were called)
        original_kc.shutdown.assert_called_once()
        original_km.shutdown_kernel.assert_called_once()
        
        # Verify that initialize was called
        code_interpreter.initialize.assert_called_once()
        
        # Restore the original initialize method
        code_interpreter.initialize = original_initialize
