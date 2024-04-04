#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 16:07
@Author  : alexanderwu
@File    : common.py
"""
import ast
import inspect
import os
import asyncio

from contextvars import ContextVar

current_session = ContextVar('current_session', default=None)

def check_cmd_exists(command) -> int:
    """Check if a command exists in the current system."""
    check_command = 'command -v ' + command + ' >/dev/null 2>&1 || { echo >&2 "no mermaid"; exit 1; }'
    result = os.system(check_command)
    return result


class UnexpectedStringOutputError(Exception):
    pass


class NoMoneyException(Exception):
    """Raised when the operation cannot be completed due to insufficient funds"""

    def __init__(self, amount, message="Insufficient funds"):
        self.amount = amount
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message} -> Amount required: {self.amount}'


def print_members(module, indent=0):
    """
    https://stackoverflow.com/questions/1796180/how-can-i-get-a-list-of-all-classes-within-current-module-in-python
    :param module:
    :param indent:
    :return:
    """
    prefix = ' ' * indent
    for name, obj in inspect.getmembers(module):
        print(name, obj)
        if inspect.isclass(obj):
            print(f'{prefix}Class: {name}')
            # print the methods within the class
            if name in ['__class__', '__base__']:
                continue
            print_members(obj, indent + 2)
        elif inspect.isfunction(obj):
            print(f'{prefix}Function: {name}')
        elif inspect.ismethod(obj):
            print(f'{prefix}Method: {name}')

class EventBus:
    """An event bus class."""

    def __init__(self, name, logger=None):
        """Initialize the event bus."""
        self._callbacks = {}
        self._logger = logger
        self.name = name

    def on(self, event_name, func):
        """Register an event callback."""
        if self._callbacks.get(event_name):
            self._callbacks[event_name].add(func)
        else:
            self._callbacks[event_name] = {func}
        return func

    def once(self, event_name, func):
        """Register an event callback that only run once."""
        if self._callbacks.get(event_name):
            self._callbacks[event_name].add(func)
        else:
            self._callbacks[event_name] = {func}
        # mark once callback
        self._callbacks[event_name].once = True
        return func

    async def aemit(self, event_name, *data):
        """Trigger an event."""
        futures = []
        for func in self._callbacks.get(event_name, []):
            try:
                if inspect.iscoroutinefunction(func):
                    futures.append(func(*data))
                else:
                    func(*data)
                if hasattr(func, "once"):
                    self.off(event_name, func)
            except Exception as e:
                if self._logger:
                    self._logger.error(
                        "Error in event callback: %s, %s, error: %s",
                        event_name,
                        func,
                        e,
                    )
        await asyncio.gather(*futures)

    def emit(self, event_name, *data):
        """Trigger an event."""
        for func in self._callbacks.get(event_name, []):
            try:
                if inspect.iscoroutinefunction(func):
                    asyncio.get_running_loop().create_task(func(*data))
                else:
                    func(*data)
                if hasattr(func, "once"):
                    self.off(event_name, func)
            except Exception as e:
                if self._logger:
                    self._logger.error(
                        "Error in event callback: %s, %s, error: %s",
                        event_name,
                        func,
                        e,
                    )

    def off(self, event_name, func=None):
        """Remove an event callback."""
        if not func:
            del self._callbacks[event_name]
        else:
            self._callbacks.get(event_name, []).remove(func)

    async def stream_callback(self, message):
        if message.type == "function_call":
            if message.status == "in_progress":
                print(message.arguments, end="", flush=True)
            else:
                print(f'\nGenerating {message.name} ({message.status}): {message.arguments}', flush=True)
        elif message.type == "text":
            print(message.content, end="", flush=True)

    def register_default_events(self):
        self.on("stream", self.stream_callback)