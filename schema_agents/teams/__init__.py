#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 00:30
@Author  : alexanderwu
@File    : software_team.py
"""
from typing import Union
from pydantic import BaseModel, Field
from schema_agents.config import CONFIG
from schema_agents.environment import Environment
from schema_agents.logs import logger
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.utils.common import NoMoneyException, EventBus


class Team(BaseModel, Role):
    """
    Team: Possesses a team, SOP (Standard Operating Procedures), and a platform for instant messaging,
    dedicated to writing executable code.
    """
    environment: Environment = Field(default_factory=Environment)
    investment: float = Field(default=10.0)
    idea: str = Field(default="")
    event_bus: EventBus = Field(default_factory=EventBus)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.environment.event_bus = self.event_bus

    def hire(self, roles: list[Role]):
        """Hire roles to cooperate"""
        self.environment.add_roles(roles)

    def invest(self, investment: float):
        """Invest team. raise NoMoneyException when exceed max_budget."""
        self.investment = investment
        CONFIG.max_budget = investment
        logger.info(f'Investment: ${investment}.')

    def _check_balance(self):
        if CONFIG.total_cost > CONFIG.max_budget:
            raise NoMoneyException(CONFIG.total_cost, f'Insufficient funds: {CONFIG.max_budget}')

    def start(self, idea: Union[str, Message]):
        """Start a project from publishing boss requirement."""
        msg = Message(role="User", content=idea) if isinstance(idea, str) else idea
        self.environment.publish_message(msg)

    def _save(self):
        logger.info(self.json())
        
    async def handle(self, message: Message):
        """Handle message"""
        self._check_balance()
        self.environment.publish_message(message)
        return await self.environment.run()

    async def run(self, n_round=1):
        """Run team until target round or no money"""
        responses = []
        while n_round > 0:
            # self._save()
            n_round -= 1
            logger.debug(f"{n_round=}")
            self._check_balance()
            responses.extend(await self.environment.run())
        return responses


if __name__ == "__main__":
    import asyncio
    team = Team()
    
    async def test(message: str):
        """Test action"""
        print(message)

    team.hire([Role(profile="QA", actions=[test])])
    asyncio.run(team.handle(Message(role="User", content="I want to build a software team.")))

    team.start("I want to build a software team.")
    asyncio.run(team.run())
    