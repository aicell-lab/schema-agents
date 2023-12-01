# -*- coding: utf-8 -*-
"""Script to create an image analyser bot

J. Metz <metz.jp@gmail.com>
"""

import io
import sys
import subprocess
import asyncio
import traceback
import imageio
import base64
import sys
import datetime
from pathlib import Path
from typing import Union, Optional

from pydantic import BaseModel, Field
import matplotlib.pyplot as plt

from schema_agents.role import Role
from schema_agents.schema import Message

from openai import AsyncOpenAI


TRIES = 5
START_TIME = datetime.datetime.now().isoformat()


class ImageAnalysisRequest(BaseModel):
    """Creating an image analysis request."""
    request: str = Field(description="This represents what the user wants to do.")
    input_image_url: str = Field(description="The url of the image file to be analysed.")
    analysis_plan: list[str] = Field(description="Steps required to fullfil the users request")


class ImageAnalysisCode(BaseModel):
    """Creates image analysis script python 3 using PEP 8 and best practices, to fullfil an analysis request"""
    script: str = Field(description="""Generated Python script for image analysis. In the script you will have access to a local variable `input_image`
                        which contains a numpy array of shape XYC which was loaded using imageio.imread. The output of the analysis should be stored
                        as a variable called `output_image`. Use print to provide extra information for debugging and error purposes.
                        """)
    dependencies: list[str]= Field(description="List of python pip packages needed by the script")


def encode_image_from_path(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


async def aask_vision(prompt, image):
    """Perform a Vision API call"""
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's "},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"}
                    },
                ]
            },
        ],
        max_tokens=300,
    )
    return response.choices[0]


def execute_code(script, context=None) -> dict:
    if context is None:
        context = {}

    # Redirect stdout and stderr to capture their output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        # Create a copy of the context to avoid modifying the original
        local_vars = context.copy()

        # Execute the provided Python script with access to context variables
        exec(script, local_vars)

        # Capture the output from stdout and stderr
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()

        return {
            "success": True,
            "stdout": stdout_output,
            "stderr": stderr_output,
            "context": local_vars  # Include context variables in the result
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "context": context  # Include context variables in the result even if an error occurs
        }
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def save_attempt(num, code="", error=None):
    filename_code = f"{START_TIME}_attempt_{num}_code.py"
    filename_error = f"{START_TIME}_attempt_{num}_error.log"
    Path(filename_code).write_text(code)
    Path(filename_error).write_text(error)


class DirectResponse(BaseModel):
    """Direct response to a user's question."""
    response: str = Field(description="The response to the user's question answering what that asked for.")


def combine_input_and_output_images(input_image_url, output_image):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    input_image = imageio.imread(input_image_url)
    ax1.imshow(input_image)
    ax2.imshow(output_image)
    fout = io.BytesIO()
    plt.savefig(fout, format="png")
    plt.close()
    fout.seek(0)
    base64_string = base64.b64encode(fout.read()).decode()
    return base64_string


def run_code_response(image_url: str, resp: ImageAnalysisCode) -> dict:
    for module in resp.dependencies:
        subprocess.run([sys.executable, "-m", "pip", "install", module])
    image = imageio.imread(image_url)
    if image.ndim == 2:
        image = image[...,None]
    assert image.shape[2] in (1,3)
    return execute_code(resp.script, {"input_image": image})


async def respond_to_user(query: str, role: Role = None) -> str:
    """Respond to user's request"""
    # response = await role.aask(query, ImageAnalysisRequest)
    resp = await role.aask(query, Union[DirectResponse, ImageAnalysisRequest])
    if isinstance(resp, DirectResponse):
        return resp.response

    if isinstance(resp, ImageAnalysisRequest):
        resp_code = await role.aask(resp, ImageAnalysisCode)
        # packages = ",".join([f"'{p}'" for p in reps.dependencies])
        ret = run_code_response(resp.input_image_url, resp_code)
        for try_number in range(TRIES):
            try:
                if not ret["success"]:
                    raise RuntimeError(f"Generated code failed to run:\n{ret['stderr']}")

                if "context" not in ret:
                    traceback.print_exc()
                    raise Exception(f"Something else happened, context is not in run_code_response return...:\n{traceback.format_exc()}")

                if "output_image" not in ret["context"]:
                    raise RuntimeError("output_image variable has not been created after running the script")
                output_image = ret["context"]["output_image"]
                if output_image.ndim not in (2,3):
                    raise RuntimeError("output_image variable does not seem to represent an image")

                composite_image = combine_input_and_output_images(resp.input_image_url, output_image)

                # Now use vision API to check output_image against the requirement
                is_good = await aask_vision(
                        f"Does the following image represent a good solution to the query: `{query}`? Give a simple answer of `yes` or `no` if you are sure, or say `unsure`",
                        composite_image)
                if "yes" not in is_good.lower():
                    raise RuntimeError("The output generated by the script does not fullfil the request")
                return resp_code
            except RuntimeError:
                # Need to refine the code, update the query

                save_attempt(try_number, code=resp_code.script, error=traceback.format_exc())

                prompt = f"""

You were asked the following: `{query}`.
You produced this script: \n`{resp_code.script}`\n,but it did not work properly, because of the following error:\n`{traceback.format_exc()}`.
Please Correct it"""
            except Exception:
                print("☠️")
                print(traceback.format_exc())
                return "Something broke 😢"

            print("New prompt:")
            print(prompt)
            resp_code = await role.aask(prompt, ImageAnalysisCode)
            ret = run_code_response(resp.input_image_url, resp_code)



    return "Could not create the required pipeline"


async def main():
    ian = Role(
        name="Ian",
        profile="ImageAnalyst",
        goal="Your goal is to listen to user's request and accept an image as in input and analyse it.",
        constraints=None,
        actions=[respond_to_user],
    )
    event_bus = ian.get_event_bus()
    event_bus.register_default_events()
    # responses = await ian.handle(Message(content="Analyse this image and tell me where the nuclei are.", role="User"))
    responses = await ian.handle(Message(content="Show me where the dark blobs in this image are : `https://imagej.net/images/blobs.gif`", role="User"))
    print("******************** Final response ********************")
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
