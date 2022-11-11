from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar, Union
from typing_extensions import Self, TypeGuard


def empty(iterable: Iterable) -> bool:
    """
    Checks if an iterable is empty.

    :param iterable: An iterable.
    :return: True if the iterable is empty, False otherwise.
    """
    return iterable is None or not any(True for _ in iterable)


_T = TypeVar("_T")
_S = TypeVar("_S")
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


def list_unique(iterable: Iterable[_T]) -> list[_T]:
    return list(dict.fromkeys(iterable))


type_cls = type


class Monad(ABC, Generic[_T, _S]):
    _value: _T

    def __init__(self, value: _T):
        self._value = value
        self._requires_simplification = True

    @property
    def value(self):
        return self._value

    @abstractmethod
    def run_single(self, transformation: Callable[[_S], Self]) -> Self:
        pass

    def run(self, *transformations: Callable[[_S], Self]) -> Self:
        result = self
        for transformation in transformations:
            result = result.run_single(transformation)
        return result

    def __rshift__(self, transformation: Callable[[_S], Self]) -> Self:
        return self.run_single(transformation)


class Nothing:
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Nothing)


nothing = Nothing()


class Option(Monad[Union[_T, Nothing], _T], Generic[_T]):
    def __new__(cls: type[Self], value: Union[_T, Nothing]) -> "Option[_T]":
        if isinstance(value, Option):
            return value
        else:
            return super().__new__(cls)

    def run_single(self, transformation: Callable[[_T], Self]) -> "Option[_T]":
        if isinstance(self.value, Nothing):
            return Option(nothing)
        return transformation(self.value)


def some(value: _T) -> Option[_T]:
    return Option(value)


x = some(5) >> (lambda x: some(x + 1)) >> (lambda x: some(x * 2))
