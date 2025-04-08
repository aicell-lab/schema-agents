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

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from transformers import is_torch_available
from transformers.testing_utils import get_tests_dir, require_torch

from schema_agents.agent_types import _AGENT_TYPE_MAPPING
from schema_agents.default_tools import FinalAnswerTool

from .test_tools import ToolTesterMixin


if is_torch_available():
    import torch


@pytest.fixture
def final_answer_tool():
    return FinalAnswerTool()


@pytest.fixture
def inputs():
    return {"answer": "Final answer"}


@pytest.mark.asyncio
async def test_exact_match_arg(final_answer_tool):
    result = await final_answer_tool("Final answer")
    assert result == "Final answer"


@pytest.mark.asyncio
async def test_exact_match_kwarg(final_answer_tool, inputs):
    result = await final_answer_tool(answer=inputs["answer"])
    assert result == "Final answer"


def create_inputs():
    inputs_text = {"answer": "Text input"}
    inputs_image = {"answer": Image.open(Path(get_tests_dir("fixtures")) / "000000039769.png").resize((512, 512))}
    inputs_audio = {"answer": torch.Tensor(np.ones(3000))}
    return {"string": inputs_text, "image": inputs_image, "audio": inputs_audio}


@require_torch
@pytest.mark.asyncio
async def test_agent_type_output(final_answer_tool):
    inputs = create_inputs()
    for input_type, input in inputs.items():
        output = await final_answer_tool(**input, sanitize_inputs_outputs=True)
        agent_type = _AGENT_TYPE_MAPPING[input_type]
        assert isinstance(output, agent_type)


def test_common_attributes(final_answer_tool):
    assert hasattr(final_answer_tool, "description")
    assert hasattr(final_answer_tool, "name")
    assert hasattr(final_answer_tool, "inputs")
    assert hasattr(final_answer_tool, "output_type")


def test_inputs_output(final_answer_tool):
    assert hasattr(final_answer_tool, "inputs")
    assert hasattr(final_answer_tool, "output_type")

    inputs = final_answer_tool.inputs
    assert isinstance(inputs, dict)

    for _, input_spec in inputs.items():
        assert "type" in input_spec
        assert "description" in input_spec
        from schema_agents.tools import AUTHORIZED_TYPES
        assert input_spec["type"] in AUTHORIZED_TYPES
        assert isinstance(input_spec["description"], str)

    output_type = final_answer_tool.output_type
    assert output_type in AUTHORIZED_TYPES
