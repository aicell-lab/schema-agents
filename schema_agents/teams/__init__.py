#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 00:30
@Author  : alexanderwu
@File    : software_team.py
"""
from typing import Union, Optional
from pydantic import BaseModel, Field 
from schema_agents.config import CONFIG
from schema_agents.logs import logger
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.utils.common import NoMoneyException, EventBus
import uuid

class Team(Role):
    """
    Team: Possesses a team, SOP (Standard Operating Procedures), and a platform for instant messaging,
    dedicated to writing executable code.
    """
    def __init__(self, *args, investment=10, **kwargs):
        super().__init__(*args, **kwargs)
        assert not kwargs.get("actions"), "Team can't have actions."
        self._investment: float = investment
        self._roles = []

    def hire(self, roles: list[Role]):
        """Hire roles to cooperate"""
        roles = list(set(roles))
        for role in roles:
            role.set_event_bus(self._event_bus)
        self._roles.extend(roles)
    
    def fire(self, role: Role):
        """Fire a role"""
        self._roles.remove(role)
        role.set_event_bus(None)

    def invest(self, investment: float):
        """Invest team. raise NoMoneyException when exceed max_budget."""
        self._investment += investment
        CONFIG.max_budget = self._investment
        logger.info(f'Investment: ${self._investment}.')

    def _check_balance(self):
        if CONFIG.total_cost > CONFIG.max_budget:
            raise NoMoneyException(CONFIG.total_cost, f'Insufficient funds: {CONFIG.max_budget}')

    def _save(self):
        logger.info(self.json())
    
    def can_handle(self, message: Message) -> bool:
        """Check if the team can handle the message."""
        if not self._roles:
            raise ValueError(f"Team {self.name} haven't hired anyone.")
        for role in self._roles:
            if role.can_handle(message):
                return True
        return False

    async def handle(self, message: Union[str, Message]) -> list[Message]:
        """Handle message"""
        if isinstance(message, str):
            message = Message(role="User", content=message)
        if not self.can_handle(message):
            raise ValueError(f"Team {self.name} can't handle message: {message}")
        _session_id = message.session_id or str(uuid.uuid4())
        message.session_history.append(_session_id)
        messages = []
        def on_message(new_msg):
            self._check_balance()
            if _session_id in new_msg.session_history:
                messages.append(new_msg)

        self._event_bus.on("message", on_message)
        try:
            # start to process the message
            await self._event_bus.aemit("message", message)
        finally:
            self._event_bus.off("message", on_message)
        return messages
        


if __name__ == "__main__":
    import asyncio
    team = Team(name="Software Team")
    
    async def test(message: str):
        """Test action"""
        print(message)

    team.hire([Role(profile="QA", actions=[test])])
    asyncio.run(team.handle(Message(role="User", content="I want to build a software team.")))
