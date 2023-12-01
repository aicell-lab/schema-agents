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
import json
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
    """Creates image analysis script using python to fullfil an analysis request"""
    # f"""Creates image analysis script python {sys.version_info.major}.{sys.version_info.minor} using PEP 8 and best practices, to fullfil an analysis request"""
    script: str = Field(description=f"""Generated Python {sys.version_info.major}.{sys.version_info.minor} script for image analysis. In the script you will have access to a local variable `input_image`
                        which contains a numpy array of shape XYC which was loaded using imageio.imread. The output of the analysis should be stored
                        as a variable called `output_image` which is a numpy array representing an image. Use print to provide extra information for debugging and error purposes.
                        """)
    dependencies: list[str]= Field(description="List of python pip packages needed by the script")


def encode_image_from_path(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


async def aask_vision(prompt, image):
    """Perform a Vision API call"""

    system_message = (
        "Act as an image analysis expert, that has been given a summary image composed of two panels."
        " The left panel shows the input image provided by the user."
        "The right panel shows the output image produced by another image analyst."
        "Your goal is to provide feedback on if the output has satisfied the request from the user. "
        "You must respond with a `valid JSON string`, starting with a `{`. No other text output is allowed."
        'Here is are example responses: `{"success": true, "details": "The output clearly satisfies the users request..."}`'
        '`{"success": false, "details": "The output does not satisfy the users request because..."}`'
    )
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"}
                    },
                ]
            },
        ],
        # response_format={ "type": "json_object" },
        max_tokens=600,
    )
    print(response.choices[0].message.content)

    raw_out = response.choices[0].message.content
    try:
        resp = json.loads(raw_out)
    except:
        resp = {"success": False, "details": raw_out}
    return resp

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
            "stacktrace": "",
            "context": local_vars  # Include context variables in the result
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "stacktrace": traceback.format_exc(),
            "context": context  # Include context variables in the result even if an error occurs
        }
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def save_attempt(num, code="", error=None, image=None):
    filename_code = f"{START_TIME}_attempt_{num}_code.py"
    filename_error = f"{START_TIME}_attempt_{num}_error.log"
    filename_image = f"{START_TIME}_attempt_{num}_image.png"
    Path(filename_code).write_text(code)
    Path(filename_error).write_text(error)
    Path(filename_image).write_bytes(base64.b64decode(image))


class DirectResponse(BaseModel):
    """Direct response to a user's question."""
    response: str = Field(description="The response to the user's question answering what that asked for.")


def combine_input_and_output_images(input_image_url, output_image):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    input_image = imageio.imread(input_image_url)
    ax1.imshow(input_image)
    ax1.set_title("Input Image")
    ax2.imshow(output_image)
    ax2.set_title("Output Image")
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
            composite_image = ''
            vision_result= {}
            try:
                if not ret["success"]:
                    raise RuntimeError(f"Generated code failed to run:\n{ret['stderr']}")

                if "context" not in ret:
                    traceback.print_exc()
                    raise Exception(f"Something else happened, context is not in run_code_response return...:\n{traceback.format_exc()}")

                if "output_image" not in ret["context"]:
                    raise RuntimeError("output_image variable has not been created after running the script")
                output_image = ret["context"]["output_image"]
                try:
                    if output_image.ndim not in (2,3):
                        raise RuntimeError("output_image variable does not seem to represent an image")
                except Exception:
                    raise RuntimeError("output_image does not appear to be a numpy array")

                composite_image = combine_input_and_output_images(resp.input_image_url, output_image)

                # Now use vision API to check output_image against the requirement
                vision_result = await aask_vision(
                        "Here is the request:"
                        f"`{resp.request}`.",
                        composite_image)
                print("VISION SAYS:", vision_result)
                if not vision_result["success"]:
                    raise RuntimeError("The output generated by the script does not fullfil the request")

                save_attempt(try_number, code=resp_code.script, error=f"{traceback.format_exc()}\n{ret.get('stacktrace','')}", image=composite_image)
                return resp_code
            except RuntimeError:
                # Need to refine the code, update the query

                save_attempt(try_number, code=resp_code.script, error=f"{traceback.format_exc()}\n{ret.get('stacktrace','')}", image=composite_image)

                prompt = f"""

You were asked the following: `{query}`.
You produced this script: \n`{resp_code.script}`\n,but it did not work properly. The issue was: `vision_result.get('details')`
\n`{traceback.format_exc()}`\n{ret.get('stacktrace','')}.
Please Correct it"""
            except Exception:
                print("☠️")
                print(traceback.format_exc())
                return "Something broke 😢"

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
    # responses = await ian.handle(Message(content="Show me where the dark blobs in this image are : `https://imagej.net/images/blobs.gif`", role="User"))
    responses = await ian.handle(Message(content="Plot the distribution of sizes of the dark blobs in the following image : `https://imagej.net/images/blobs.gif`", role="User"))
    print("******************** Final response ********************")
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
