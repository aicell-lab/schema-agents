#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/20 12:15
@Author  : alexanderwu
@File    : memory.py
"""
from collections import defaultdict, OrderedDict
from typing import Iterable, Callable
from pydantic import BaseModel

from schema_agents.schema import Message


class Memory:
    """The most basic memory: super-memory"""

    def __init__(self):
        """Initialize an empty storage list and an empty index dictionary"""
        self.storage: list[Message] = []
        self.index: OrderedDict[Callable, list[Message]] = OrderedDict()

    def add(self, message: Message):
        """Add a new message to storage, while updating the index"""
        if message in self.storage:
            return
        self.storage.append(message)
        if message.cause_by:
            if message.cause_by not in self.index:
                self.index[message.cause_by] = []
            self.index[message.cause_by].append(message)


    def add_batch(self, messages: Iterable[Message]):
        for message in messages:
            self.add(message)

    def get_by_role(self, role: str) -> list[Message]:
        """Return all messages of a specified role"""
        return [message for message in self.storage if message.role == role]

    def get_by_content(self, content: str) -> list[Message]:
        """Return all messages containing a specified content"""
        return [message for message in self.storage if content in message.content]
    
    def get_by_schemas(self, schemas: list[BaseModel]) -> list[Message]:
        ret = []
        for schema in schemas:
            if schema is str:
                ret += [message for message in self.storage if not message.data and isinstance(message.content, str)]
            else:
                ret += [message for message in self.storage if message.data and isinstance(message.data, schema)]
        return ret

    def delete(self, message: Message):
        """Delete the specified message from storage, while updating the index"""
        self.storage.remove(message)
        if message.cause_by and message in self.index[message.cause_by]:
            self.index[message.cause_by].remove(message)

    def clear(self):
        """Clear storage and index"""
        self.storage = []
        self.index = defaultdict(list)

    def count(self) -> int:
        """Return the number of messages in storage"""
        return len(self.storage)

    def try_remember(self, keyword: str) -> list[Message]:
        """Try to recall all messages containing a specified keyword"""
        return [message for message in self.storage if keyword in message.content]

    def get(self, k=0) -> list[Message]:
        """Return the most recent k memories, return all when k=0"""
        return self.storage[-k:]

    def remember(self, observed: list[Message], k=10) -> list[Message]:
        """remember the most recent k memories from observed Messages, return all when k=0"""
        already_observed = self.get(k)
        news: list[Message] = []
        for i in observed:
            if i in already_observed:
                continue
            news.append(i)
        return news

    def get_by_action(self, action: Callable) -> list[Message]:
        """Return all messages triggered by a specified Action"""
        return self.index[action]

    def get_by_actions(self, actions: Iterable[Callable]) -> list[Message]:
        """Return all messages triggered by specified Actions"""
        rsp = []
        for key in self.index.keys():
            if key in actions:
                rsp += self.index[key]
        return rsp
