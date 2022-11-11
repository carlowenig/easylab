from typing import Any, Literal, TypeGuard, Union

EllipsisType = type(Ellipsis)

Wildcard = Union[EllipsisType, Literal["*"]]


def is_wildcard(value: Any) -> TypeGuard[Wildcard]:
    return value is Ellipsis or (isinstance(value, str) and value == "*")
