#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 22:12
@Author  : alexanderwu
@File    : environment.py
"""
import asyncio
from typing import Iterable

from pydantic import BaseModel, Field

from schema_agents.memory import Memory
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.utils.common import EventBus

class Environment(BaseModel):
    """环境，承载一批角色，角色可以向环境发布消息，可以被其他角色观察到"""

    roles: dict[str, Role] = Field(default_factory=dict)
    memory: Memory = Field(default_factory=Memory)
    history: str = Field(default='')
    user_support_roles: set[Role] = Field(default_factory=set)
    event_bus: EventBus = None

    class Config:
        arbitrary_types_allowed = True

    def add_role(self, role: Role):
        """增加一个在当前环境的Role"""
        role.set_env(self)
        self.roles[role.profile] = role
        if role.user_support_actions:
            self.user_support_roles.add(role)
        if self.event_bus is not None:
            self.event_bus.emit("role_add", role)

    def add_roles(self, roles: Iterable[Role]):
        """增加一批在当前环境的Role"""
        for role in roles:
            self.add_role(role)

    def publish_message(self, message: Message):
        """向当前环境发布信息"""
        # self.message_queue.put(message)
        self.memory.add(message)
        self.history += f"\n{message}"
        if self.event_bus is not None:
            self.event_bus.emit("message", message)

    async def run(self):
        """处理一次所有Role的运行"""
        # while not self.message_queue.empty():
        # message = self.message_queue.get()
        # rsp = await self.manager.handle(message, self)
        # self.message_queue.put(rsp)
        futures = []
        for role in self.roles.values():
            future = role.run()
            futures.append(future)

        responses = await asyncio.gather(*futures)
        if self.event_bus is not None:
            self.event_bus.emit("run", responses)
        return responses

    def get_roles(self) -> dict[str, Role]:
        """获得环境内的所有Role"""
        return self.roles

    def get_role(self, name: str) -> Role:
        """获得环境内的指定Role"""
        return self.roles.get(name, None)
