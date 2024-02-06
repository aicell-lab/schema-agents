#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from schema_agents.role import Role
from schema_agents.schema import Message
import numpy as np

    
@pytest.mark.asyncio
async def test_logprobs():
    async def respond_to_user(query: str, role: Role) -> str:
        """Respond to user."""
        messages = [{"role": "user", "content": "How are you today? Anwser YES or NO, nothing else!"}]
        text, logprops = await role.llm.acompletion(messages, logprobs=True, top_logprobs=3)
        logprobs_bytes = []
        for prob in logprops:
            logprobs_bytes.extend(prob.bytes)  # Use extend instead of +=
        text_response = bytes(logprobs_bytes).decode("utf-8")
        assert text == text_response
        assert logprops[0].token == 'YES'
        assert np.round(np.exp(logprops[0].logprob),2) > 0.5

    alice = Role(
        name="Alice",
        profile="Cooker",
        goal="Your goal is to listen to user's request and propose recipes for making the most delicious meal for thanksgiving.",
        constraints=None,
        actions=[respond_to_user],
        model="gpt-4-turbo-preview",
    )
    event_bus = alice.get_event_bus()
    event_bus.register_default_events()
    await alice.handle(Message(content="make something to surprise our guest from Stockholm.", role="User"))
    