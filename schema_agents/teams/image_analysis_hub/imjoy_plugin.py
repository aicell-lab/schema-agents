from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import asyncio
import json
from functools import partial
from schema_agents.role import Role, Action
from schema_agents.schema import Message

class ImJoyPluginConfig(BaseModel):
    """ImJoy Plugin Configuration"""
    name: str = Field(..., max_length=48, description="Name of the plugin, must be unique")
    type: str = Field(..., description="Plugin type, e.g., web-worker, window, native-python, or web-python")
    version: str = Field("0.1.0", max_length=32, description="Version of the plugin")
    api_version: str = Field("0.1.8", max_length=32, description="API version of ImJoy the plugin is written for")
    description: str = Field(..., max_length=256, description="Short description of the plugin")
    icon: str = Field(default="extension", description="Name of the icon an emoji icon or an image URL")
    permissions: List[str] = Field(default=[], description="List of browser iframe permissions needed by the plugin")
    requirements: List[str] = Field(default=[], description="Requirements specification for the plugin, e.g., pip package names for python or url for javascript/css")
    dependencies: List[str] = Field(default=[], description="List of names of dependent plugin urls")
    labels: List[str] = Field(default=[], max_length=32, description="Labels associated with the plugin")
    # defaults: dict = Field(default={}, description="Default settings for window plugins")
    # inputs: Union[dict, list, None] = Field(default=None, description="Input schema for matching files/data to the plugin")
    # outputs: Union[dict, list, None] = Field(default=None, description="Output schema for the plugin results")
    # env: str = Field(default=None, description="Virtual environment or Docker image for running the plugin")
    # base_frame: Optional[str] = Field(default=None, description="Custom HTML page to load in window plugins")
    # runnable: bool = Field(default=True, description="Whether the plugin can be executed from the menu")
    # allow_execution: bool = Field(..., description="Boolean value for allowing execution")
    # auth: Union[dict, None] = Field(default=None, description="Authentication configuration")
    # cover: Union[str, List[str], None] = Field(default=None, max_length=1024, description="Cover image or images for the plugin")
    # dedicated_thread: bool = Field(default=False, description="Whether the plugin runs in a dedicated thread")
    # flags: List[str] = Field(default=[], max_length=32, description="Array of flags for the plugin")
    # id: str = Field(..., max_length=128, description="Unique ID for the plugin")
    # lang: str = Field(default=None, max_length=32, description="Language specification for the plugin")
    # tags: List[str] = Field(default=[], max_length=32, description="Tags associated with the plugin")
    # ui: str = Field(default=None, max_length=2048, description="UI configuration for the plugin")
    # docs: Union[str, dict, None] = Field(default=None, description="Documentation for the plugin")

class ImJoyPluginFile(BaseModel):
    """ImJoy Plugin File"""
    config: ImJoyPluginConfig = Field(..., description="The plugin configuration")
    script: str = Field(..., description="A code block in JavaScript or Python")
    script_lang: str = Field(..., description="The language of the script, either `javascript` or `python`")
    window: Optional[str] = Field(None, description="A code block in HTML format for plugins in `window` mode")
    style: Optional[str] = Field(None, description="A code block in CSS format for plugins in `window` mode")
    docs: Optional[str] = Field(None, description="A recommended code block in Markdown format with the documentation of the plugin")
    attachment: Optional[Dict[str, str]] = Field(None, description="Optional blocks for storing text data, the key is attachment name and the value is the text content")

def format_imjoy_plugin(plugin: ImJoyPluginFile):
    src = f'<config lang="json">\n{json.dumps(plugin.config.dict())}\n</config>'
    src += f'\n<script lang="{plugin.script_lang}">\n{plugin.script}\n</script>'
    if plugin.window:
        src += f'\n<window>\n{plugin.window}\n</window>'
    if plugin.style:
        src += f'\n<style>\n{plugin.style}\n</style>'
    if plugin.docs:
        src += f'\n<docs>\n{plugin.docs}\n</docs>'
    if plugin.attachment:
        for name, content in plugin.attachment.items():
            src += f'\n<attachment name="{name}">\n{content}\n</attachment>'
    return src

async def create_imjoy_plugin(config: ImJoyPluginFile) -> str:
    """Create an imjoy plugin file"""
    plugin_src = format_imjoy_plugin(config)
    # plugin = await client.load_plugin(src=plugin_src)
    # await plugin.run()
    return plugin_src

# original_requirements="The user want to create an imjoy plugin for load an image file, blur it with a button and download the result.",
# user_interface="The user interface contains an upload button, process button, a gaussian kernel number input, two image tag side by side.",
# user_interaction="The user click a button to upload the image, then select a file, once selected the image get displayed on in the plugin, then the user click the process button, the image will be blurred with the selected gaussian kernel, the result get displayed on the side.",
# external_api_calls={
#     "blur()": "for blurring the image",
#     "download()": "for downloading the result",
# },
# anything_unclear="none",

class ImJoyPluginRequirement(BaseModel):
    """ImJoy Plugin Requirement"""
    original_requirements: str = Field(description="The polished complete original requirements")
    user_interface: str = Field(description="How the user interface looks like")
    user_interaction: str = Field(description="How the user interact with the plugin interface")
    external_api_calls: Dict[str, str] = Field(description="The list of external imjoy plugin api calls")
    anything_unclear: str = Field(description="Anything unclear")

if __name__ == "__main__":
    ImJoyPluginAction = Action.create(create_imjoy_plugin)
    ImJoyPluginRequirementAction = Action.create(ImJoyPluginRequirement)
    PluginDeveloper = Role.create(name="Bob",
        profile="Plugin Developer",
        goal="Develop ImJoy Plugin based on user's requirements.",
        constraints=None,
        actions=[ImJoyPluginAction],
        watch=[ImJoyPluginRequirementAction])
    dev = PluginDeveloper()
    req = ImJoyPluginRequirement(
        original_requirements="The user want to create an imjoy plugin for load an image file, blur it with a button and download the result.",
        user_interface="The user interface contains an upload button, process button, a gaussian kernel number input, two image tag side by side.",
        user_interaction="The user click a button to upload the image, then select a file, once selected the image get displayed on in the plugin, then the user click the process button, the image will be blurred with the selected gaussian kernel, the result get displayed on the side.",
        external_api_calls={
            "processor.blur": "for blurring the image",
            "api.export": "for downloading the result",
        },
        anything_unclear="none",
    )
    msg = Message(content=req.json(), instruct_content=req, role="Boss", cause_by=ImJoyPluginRequirementAction)
    dev.recv(msg)
    assert msg in dev._rc.important_memory
    asyncio.run(dev._react())
    assert ImJoyPluginRequirementAction in dev._rc.important_memory