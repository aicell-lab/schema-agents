"""Team definitions for schema-agents."""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field, ConfigDict
from schema_agents.role import Role

class Team(BaseModel):
    """Schema for agent teams."""
    name: str = Field(..., description="Team name")
    description: str = Field(..., description="Team description")
    roles: List[Role] = Field(..., description="Team roles")
    leader_role: Optional[str] = Field(None, description="Team leader role name")
    shared_knowledge: Dict[str, Any] = Field(default_factory=dict, description="Shared team knowledge")
    communication_rules: List[str] = Field(default_factory=list, description="Team communication rules")
    model_config = ConfigDict(extra="allow")

    def add_role(self, role: Role) -> None:
        """Add a role to the team."""
        if role not in self.roles:
            self.roles.append(role)

    def remove_role(self, role_name: str) -> None:
        """Remove a role from the team."""
        self.roles = [r for r in self.roles if r.name != role_name]
        if self.leader_role == role_name:
            self.leader_role = None

    def set_leader(self, role_name: str) -> None:
        """Set team leader role."""
        if any(r.name == role_name for r in self.roles):
            self.leader_role = role_name
        else:
            raise ValueError(f"Role {role_name} not found in team")

    def get_role(self, role_name: str) -> Optional[Role]:
        """Get a role by name."""
        return next((r for r in self.roles if r.name == role_name), None)

    def share_knowledge(self, key: str, value: Any) -> None:
        """Share knowledge across team."""
        self.shared_knowledge[key] = value
        for role in self.roles:
            role.update_knowledge(key, value)

    def get_shared_knowledge(self, key: str) -> Optional[Any]:
        """Get shared team knowledge."""
        return self.shared_knowledge.get(key)

    def get_capabilities(self) -> Set[str]:
        """Get all team capabilities."""
        return set().union(*(set(role.capabilities) for role in self.roles))

    def validate_action(self, action: str, role_name: Optional[str] = None) -> bool:
        """Validate if an action is within team capabilities."""
        if role_name:
            role = self.get_role(role_name)
            return role is not None and role.validate_action(action)
        return any(role.validate_action(action) for role in self.roles)

    def merge_team(self, other: Team) -> Team:
        """Merge with another team."""
        return Team(
            name=f"{self.name} + {other.name}",
            description=f"Combined team of {self.name} and {other.name}",
            roles=list(set(self.roles + other.roles)),
            leader_role=None,  # Reset leader in merged team
            shared_knowledge={**self.shared_knowledge, **other.shared_knowledge},
            communication_rules=list(set(self.communication_rules + other.communication_rules))
        ) 