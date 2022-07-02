from typing import Any, Callable, Iterable, TypeVar, Union
from typing_extensions import TypeGuard


def empty(iterable: Iterable) -> bool:
    """
    Checks if an iterable is empty.

    :param iterable: An iterable.
    :return: True if the iterable is empty, False otherwise.
    """
    return not any(True for _ in iterable)


_T = TypeVar("_T")
_R = TypeVar("_R")


def collect_args(f: Callable[[tuple], _R]) -> Callable[..., _R]:
    def wrapper(*args) -> _R:
        return f(args)

    return wrapper


def collect_args_method(f: Callable[[Any, tuple], _R]) -> Callable[..., _R]:
    def wrapper(self, *args) -> _R:
        return f(self, args)

    return wrapper


def all_are_instance(iterable: Iterable, _type: type[_T]) -> TypeGuard[Iterable[_T]]:
    return all(isinstance(item, _type) for item in iterable)


def all_fulfill_type_guard(
    iterable: Iterable, type_guard: Callable[[Any], TypeGuard[_T]]
) -> TypeGuard[Iterable[_T]]:
    return all(type_guard(item) for item in iterable)
