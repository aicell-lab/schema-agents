#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/8 22:12
@Author  : alexanderwu
@File    : schema.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, Callable

from pydantic import BaseModel

from schema_agents.logs import logger


class RawMessage(TypedDict):
    content: str
    role: str


@dataclass
class Message:
    """list[<role>: <content>]"""
    content: str
    data: BaseModel = field(default=None)
    role: str = field(default='user')  # system / user / assistant
    cause_by: Callable = field(default=None)
    processed_by: set['Role'] = field(default_factory=set)
    session_ids: list[str] = field(default_factory=list)

    def __str__(self):
        return f"{self.role}: {self.content}"

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content
        }


@dataclass
class UserMessage(Message):
    """便于支持OpenAI的消息"""
    def __init__(self, content: str):
        super().__init__(content, 'user')


@dataclass
class SystemMessage(Message):
    """便于支持OpenAI的消息"""
    def __init__(self, content: str):
        super().__init__(content, 'system')


@dataclass
class AIMessage(Message):
    """便于支持OpenAI的消息"""
    def __init__(self, content: str):
        super().__init__(content, 'assistant')


@dataclass
class MemoryChunk:
    """list[<role>: <content>]"""
    index: str
    content: BaseModel = field(default=None)
    category: str = field(default=None)

    def __str__(self):
        return f"memory chunk: {self.index}"

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        return {
            "index": self.index
        }


if __name__ == '__main__':
    test_content = 'test_message'
    msgs = [
        UserMessage(test_content),
        SystemMessage(test_content),
        AIMessage(test_content),
        Message(test_content, role='QA')
    ]
    logger.info(msgs)
