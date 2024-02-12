#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/8 22:12
@Author  : alexanderwu
@File    : schema.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypedDict, Callable

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
    session_id: str = field(default=None)
    session_history: list[str] = field(default_factory=list)

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

class RoleSetting(BaseModel):
    """Role setting"""
    name: str
    profile: str
    goal: str
    instructions: str
    constraints: Optional[str] = ""
    icon: Optional[str] = None

    def __str__(self):
        return f"{self.name}({self.profile})"

    def __repr__(self):
        return self.__str__()


class Session(BaseModel):
    id: Optional[str] = None
    role_setting: Optional[RoleSetting] = None


class StreamEvent(BaseModel):
    type: str
    query_id: str
    session: Optional[Session] = None
    status: str
    content: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None


if __name__ == '__main__':
    test_content = 'test_message'
    msgs = [
        UserMessage(test_content),
        SystemMessage(test_content),
        AIMessage(test_content),
        Message(test_content, role='QA')
    ]
    logger.info(msgs)
