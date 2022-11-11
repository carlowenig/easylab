from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar, cast, Self

T = TypeVar("T")


class Field(ABC, Generic[T]):
    @staticmethod
    def infer_from_type(type_: type[T]) -> Field[T]:
        if issubclass(type_, int):
            return cast(Field[T], int_field)
        elif issubclass(type_, float):
            return cast(Field[T], float_field)
        else:
            raise TypeError(f"Cannot infer field for type {type_.__name__}.")

    def __init__(self, type_: type[T], zero: T, one: T) -> None:
        self.type = type_
        self.zero = zero
        self.one = one

    @abstractmethod
    def sum(self, a: T, b: T) -> T:
        pass

    @abstractmethod
    def mul(self, a: T, b: T) -> T:
        pass


class SupportsFieldOperations(Protocol):
    def __add__(self, other: Self, /) -> Self:
        ...

    def __mul__(self, other: Self, /) -> Self:
        ...


F = TypeVar("F", bound=SupportsFieldOperations)


class OperatorField(Field[F]):
    def sum(self, a: F, b: F) -> F:
        return a + b

    def mul(self, a: F, b: F) -> F:
        return a * b


int_field = OperatorField(int, 0, 1)
float_field = OperatorField(float, 0.0, 1.0)
