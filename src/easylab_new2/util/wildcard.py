from .misc import EllipsisType
from typing import Any, Literal, Union
from typing_extensions import TypeGuard


Wildcard = Union[EllipsisType, Literal["*"]]


def is_wildcard(value: Any) -> TypeGuard[Wildcard]:
    return value is Ellipsis or (isinstance(value, str) and value == "*")
