#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Any

from pydantic import BaseModel


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
    event_bus: Optional[Any] = None
    role_setting: Optional[RoleSetting] = None
    stop: bool = False
    
    def model_dump(self, **kwargs):
        # Exclude event_bus from serialization since it's not JSON serializable
        kwargs.setdefault('exclude', set()).add('event_bus')
        return super().model_dump(**kwargs)


class StreamEvent(BaseModel):
    type: str
    query_id: str
    session: Optional[Session] = None
    status: str
    content: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None

    class Config:
        # This ensures proper JSON serialization
        json_encoders = {
            Session: lambda v: v.model_dump(mode="json") if v else None
        }
