# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import pytest

from schema_agents import DuckDuckGoSearchTool
from schema_agents.agent_types import _AGENT_TYPE_MAPPING
from schema_agents.tools import AUTHORIZED_TYPES

from .test_tools import create_inputs
from .utils.markers import require_run_all


@pytest.fixture
def search_tool():
    tool = DuckDuckGoSearchTool()
    tool.setup()
    return tool


def test_common_attributes(search_tool):
    assert hasattr(search_tool, "description")
    assert hasattr(search_tool, "name")
    assert hasattr(search_tool, "inputs")
    assert hasattr(search_tool, "output_type")


def test_inputs_output(search_tool):
    assert hasattr(search_tool, "inputs")
    assert hasattr(search_tool, "output_type")

    inputs = search_tool.inputs
    assert isinstance(inputs, dict)

    for _, input_spec in inputs.items():
        assert "type" in input_spec
        assert "description" in input_spec
        assert input_spec["type"] in AUTHORIZED_TYPES
        assert isinstance(input_spec["description"], str)

    output_type = search_tool.output_type
    assert output_type in AUTHORIZED_TYPES


@pytest.mark.asyncio
@require_run_all
async def test_exact_match_arg(search_tool):
    result = await search_tool("Agents")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_agent_type_output(search_tool):
    inputs = create_inputs(search_tool.inputs)
    output = await search_tool(**inputs, sanitize_inputs_outputs=True)
    if search_tool.output_type != "any":
        agent_type = _AGENT_TYPE_MAPPING[search_tool.output_type]
        assert isinstance(output, agent_type)
