#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 00:30
@Author  : alexanderwu
@File    : software_team.py
"""
from pydantic import BaseModel, Field
from schema_agents.config import CONFIG
from schema_agents.environment import Environment
from schema_agents.logs import logger
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.utils.common import NoMoneyException


class Team(BaseModel):
    """
    Team: Possesses a team, SOP (Standard Operating Procedures), and a platform for instant messaging,
    dedicated to writing executable code.
    """
    environment: Environment = Field(default_factory=Environment)
    investment: float = Field(default=10.0)
    idea: str = Field(default="")

    class Config:
        arbitrary_types_allowed = True

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

    def start(self, idea, message_callback: callable = None):
        """Start a project from publishing boss requirement."""
        self.idea = idea
        self.environment.message_callback = message_callback
        self.environment.publish_message(Message(role="User", content=idea))

    def _save(self):
        logger.info(self.json())

    async def run(self, n_round=3):
        """Run team until target round or no money"""
        while n_round > 0:
            # self._save()
            n_round -= 1
            logger.debug(f"{n_round=}")
            self._check_balance()
            await self.environment.run()
        return self.environment.history
