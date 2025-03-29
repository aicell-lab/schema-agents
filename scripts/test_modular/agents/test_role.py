from pydantic import BaseModel, Field
from schema_agents.role import Role
from schemas import Type1, Type2, Type3

class TestRole(Role):
    """A test role"""
    def __init__(self):
        super().__init__(
            name="TestRole",
            profile="A test role",
            goal="To test stuff",
            constraints=None,
            actions=[self.fun_1, self.fun_2, self.fun_3],
        )

    async def fun_1(self, user_input: str, role: Role = None) -> Type1:
        """Test function 1"""
        result = Type1(a = "a")
        return(result)

    async def fun_2(self, type_1: Type1, role: Role = None) -> Type2:
        """Test function 2"""
        result = Type2(b = "b")
        return(result)

    async def fun_3(self, input_1 : Type1, role: Role = None) -> Type3:
        """Test function 3"""
        result = await role.aask(input_1, Type3)
        return(result)