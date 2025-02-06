#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import os
import asyncio
import uuid

from contextvars import ContextVar
from schema_agents.schema import Session

current_session = ContextVar('current_session', default=None)


from contextlib import asynccontextmanager
from contextvars import copy_context

@asynccontextmanager
async def create_session_context(id=None, role_setting=None, event_bus=None):
    id = id or str(uuid.uuid4())
    pre_session = current_session.get()
    if pre_session:
        id = id or pre_session.id
        role_setting = role_setting or pre_session.role_setting
        event_bus = event_bus or pre_session.event_bus
    current_session.set(Session(id=id, role_setting=role_setting, event_bus=event_bus))
    yield copy_context()
    current_session.set(pre_session)

def check_cmd_exists(command) -> int:
    """Check if a command exists in the current system."""
    check_command = 'command -v ' + command + ' >/dev/null 2>&1 || { echo >&2 "no mermaid"; exit 1; }'
    result = os.system(check_command)
    return result

class EventBus:
    """An event bus class."""

    def __init__(self, name, logger=None):
        """Initialize the event bus."""
        self._callbacks = {}
        self._logger = logger
        self.name = name
        self._tasks = set()  # Add this line to store task references

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
                    task = asyncio.get_running_loop().create_task(func(*data))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)  # Remove task when done
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