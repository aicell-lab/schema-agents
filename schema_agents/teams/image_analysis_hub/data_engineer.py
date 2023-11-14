import asyncio
import json
import traceback
from typing import List, Union

from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.tools.code_interpreter import create_mock_client

from schema_agents.teams.image_analysis_hub.schemas import (PythonFunctionScript, PythonFunctionScriptChanges, Change,
                      PythonFunctionScriptWithLineNumber, SoftwareRequirement)

DEPLOY_SCRIPT = """
import asyncio
from imjoy_rpc.hypha import connect_to_server

{function_code}

if not 'hypha_server' in vars() or 'hypha_server' in globals():
    hypha_server = await connect_to_server(
        {{"name": "test client", "server_url": "https://ai.imjoy.io/", "method_timeout": 3000}}
    )

service_config = {{
    "name": "Hypha",
    "id": "{service_id}",
    "config": {{
        "visibility": "public"
    }},
}}

for function_name in {function_names}:
    service_config[function_name] = globals()[function_name]

await hypha_server.register_service(service_config)
"""


def add_line_number(script):
    return "\n".join(
        [f"{i+1:03d}→{line}" for i, line in enumerate(script.split("\n"))]
    )

def apply_changes(original_script: str, changes: List[Change]) -> str:
    original_script_lines = original_script.split("\n")
    line_offset = 0  # Offset to adjust line numbers after add/remove operations

    # Sort changes by line number
    sorted_changes = sorted(changes, key=lambda x: x.line)

    for change in sorted_changes:
        adjusted_line_number = (
            change.line - 1 + line_offset
        )  # Convert 1-based line number to 0-based index and adjust

        if change.operation == "remove":
            del original_script_lines[adjusted_line_number]
            line_offset -= 1  # Decrease offset due to removal
        elif change.operation == "add":
            original_script_lines.insert(adjusted_line_number, change.content)
            line_offset += 1  # Increase offset due to addition
        elif change.operation == "replace":
            original_script_lines[adjusted_line_number] = change.content

    updated_script = "\n".join(original_script_lines)
    return updated_script


async def fix_code(
    role: Role, client, python_function: PythonFunctionScript, output_summary
):
    prompt = f"Fix the code according to the error message:\n{output_summary}\n"
    "You should call the `PythonFunctionScriptChanges` function to make changes for fixing the error,"
    " optionally, if too many changes needed, call `PythonFunctionScript` to do a compete rewrite. DO NOT return text directly."
    "Importantly, DO NOT change the test script unless the error is caused by the test script."
    python_function_with_line_number = PythonFunctionScriptWithLineNumber(
        function_script_with_line_number=add_line_number(
            python_function.function_script
        ),
        pip_packages=python_function.pip_packages,
        test_script_with_line_number=add_line_number(python_function.test_script),
        docstring=python_function.docstring,
    )
    response = await role.aask(
        python_function_with_line_number,
        Union[PythonFunctionScriptChanges, PythonFunctionScript],
        prompt=prompt,
    )
    if isinstance(response, PythonFunctionScript):
        return response
    changes = response
    try:
        patched_test_script = (
            apply_changes(python_function.test_script, changes.test_script_changes)
            if changes.test_script_changes
            else python_function.test_script
        )
    except Exception:
        output_summary += (
            f"\nFailed to apply the test_script_changes:\n{changes.test_script_changes}\nTraceback: "
            + traceback.format_exc()
        )
        return await fix_code(role, client, python_function, output_summary)

    try:
        patched_function_script = (
            apply_changes(
                python_function.function_script, changes.function_script_changes
            )
            if changes.function_script_changes
            else python_function.function_script
        )
    except Exception:
        output_summary += (
            f"\nFailed to apply the function_script_changes:\n{changes.function_script_changes}\nTraceback: "
            + traceback.format_exc()
        )
        return await fix_code(role, client, python_function, output_summary)

    return PythonFunctionScript(
        function_names=python_function.function_names,
        docstring=python_function.docstring,
        function_script=patched_function_script,
        pip_packages=changes.updated_pip_packages,
        test_script=patched_test_script,
    )


async def generate_code(
    req: SoftwareRequirement, role: Role
) -> PythonFunctionScript:
    return await role.aask(req, PythonFunctionScript)

INSTALL_SCRIPT = """
try:
    import pyodide
    import micropip
    await micropip.install([{packages}])
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', {packages}])
"""

async def test_run_python_function(
    role, client, service_id, python_function: PythonFunctionScript
) -> PythonFunctionScript:
    """Test run the python function script."""
    if python_function.pip_packages:
        packages = ",".join([f"'{p}'" for p in python_function.pip_packages])
        results = await client.executeScript({"script": INSTALL_SCRIPT.format(packages=packages)})
        output_summary = json.dumps(
            {k: results[k] for k in results.keys() if results[k]}, indent=1
        )
        if results["status"] != "ok":
            raise RuntimeError(f"Failed to install pip packages: {python_function.pip_packages}, error: {output_summary}")
    results = await client.executeScript(
        {"script": python_function.function_script + "\n" + python_function.test_script}
    )
    assert results, "No results returned from the test script."
    if results["status"] != "ok":
        output_summary = json.dumps(
            {k: results[k] for k in results.keys() if results[k]}, indent=1
        )
        python_function = await fix_code(role, client, python_function, output_summary)
        return await test_run_python_function(role, client, service_id, python_function)
    else:
        # deploy the functions
        results = await client.executeScript(
            {"script": DEPLOY_SCRIPT.format(function_names=python_function.function_names,
                                 service_id=service_id,
                                 function_code=python_function.function_script
                                )
            }
        )
    return python_function

def create_data_engineer(client=None):
    async def develop_python_functions(
        req: SoftwareRequirement, role: Role
    ) -> PythonFunctionScript:
        """Develop python functions based on software requirements."""
        # if isinstance(req, SoftwareRequirement):
        func = await generate_code(req, role)
        try:
            func = await test_run_python_function(role, client, req.id, func)
        except RuntimeError as exp:
            req.additional_notes += f"\nPlease avoid the following error: {exp}"
            func = await generate_code(req, role)
            func = await test_run_python_function(role, client, req.id, func)
        return func
        # else:
        #     if client:
        #         await client.showDialog(
        #             src="https://gist.githubusercontent.com/oeway/b734c35f69a0ec0dcebe00b078676edb/raw/react-ui-plugin.imjoy.html",
        #             data={"jsx_script": req.jsx_script}
        #         )
        #     else:
        #         print("Unable to show the React UI")
        #     return func

    data_engineer = Role(
        name="Alice",
        profile="Data Engineer",
        goal="Develop the python function script according to the software requirement, ensuring that it fulfills the desired functionality. Implement necessary algorithms, handle data processing, and write tests to validate the correctness of the function.",
        constraints=None,
        actions=[develop_python_functions],
    )
    return data_engineer


async def main():
    mock_software_requirements = {
        "id": "cell_segmentation",
        "original_requirements": "Load an image file and segment the cells in the image, count the cells then show the result image. The cells are U2OS cells in a IF microscopy image, cells are round and in green color, the background is black. For testing, you can use /Users/wei.ouyang/workspace/LibreChat/chatbot/tests/data/img16.png",
        "python_function_requirements": [
            {
                "function_name": "load_image",
                "function_signature": "load_image(image_path: str) -> np.ndarray",
                "feature_instructions": "This function should load an image from the provided path and return it as a numpy array. Use the OpenCV library for this.",
                "context_requirements": "The function will be used to load an image from a file path.",
                "testing_requirements": "Test the function with different image file paths to ensure it can handle different image formats and sizes.",
            },
            {
                "function_name": "segment_cells",
                "function_signature": "segment_cells(image: np.ndarray) -> Tuple[np.ndarray, int]",
                "feature_instructions": "This function should segment the cells in the image and return the segmented image and the count of the cells. Use the OpenCV library for the segmentation. The cells are round and in green color, the background is black.",
                "context_requirements": "The function will be used to segment the cells in the image and count them.",
                "testing_requirements": "Test the function with different images to ensure it can accurately segment and count the cells.",
            },
        ],
        "react_ui_requirements": {
            "plugin_name": "CellCounter",
            "ui_layout": "The UI should have an input field to enter the image file path, a button to load and segment the image, and a display area to show the segmented image and the cell count.",
            "interaction_patterns": "The user enters the image file path and clicks the button to load and segment the image. The segmented image and the cell count are then displayed.",
            "functionalities": "The UI should allow the user to enter the image file path, load and segment the image, and display the segmented image and the cell count.",
            "user_flows": "The user enters the image file path, clicks the button to load and segment the image, and sees the segmented image and the cell count.",
        },
        "additional_notes": "The cells are U2OS cells in a IF microscopy image. The cells are round and in green color, the background is black. The number of cells in the image should be more than 12.",
    }

    DataEngineer = create_data_engineer(client=create_mock_client())
    ds = DataEngineer()

    pr = SoftwareRequirement.parse_obj(mock_software_requirements)
    req = Message(
        content=pr.json(),
        data=pr,
        role="Project Manager",
    )

    event_bus = ds.get_event_bus()
    event_bus.register_default_events()
    messages = await ds.handle(req)
    print(messages)


if __name__ == "__main__":
    asyncio.run(main())
