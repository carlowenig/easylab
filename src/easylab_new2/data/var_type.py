from __future__ import annotations
import math
from typing import Any, Callable, Generic, TypeVar, Union
from ..lang import Text
from ..util import Undefined, undefined
import inflection


T = TypeVar("T")

VarTypeLike = Union["VarType[T]", type[T]]


class VarType(Generic[T]):
    @staticmethod
    def interpret(input: VarTypeLike[T]) -> VarType[T]:
        if isinstance(input, VarType):
            return input
        elif isinstance(input, type):
            return VarType(input)
        else:
            raise TypeError(f"Cannot interpret {input} as VarType.")

    def __init__(
        self,
        value_type: type[T],
        name: str | None = None,
        *,
        default: Callable[[], Any] | None = None,
        format: Callable[[T], Any] | None = None,
        parse: Callable[[Any], T] | None = None,
        equal: Callable[[T, T], bool] | None = None,
        check: Callable[[T], None] | None = None,
        bases: list[VarType[T]] = [],
    ) -> None:
        self.value_type = value_type
        self.name = name or inflection.underscore(value_type.__name__)
        self._default = default
        self._format = format
        self._parse = parse
        self._equal = equal
        self._check = check
        self.bases = bases

    def default_or_raise(self) -> T:
        # Own implementation
        if self._default is not None:
            return self.parse(self._default())

        # Fall back to base implementation
        for base in self.bases:
            try:
                return base.default_or_raise()
            except NotImplementedError:
                pass

        raise NotImplementedError

    def default(self) -> T | Undefined:
        try:
            return self.default_or_raise()
        except NotImplementedError:
            return undefined

    def format_or_raise(self, value: T) -> Text:
        # Own implementation
        if self._format is not None:
            return Text.interpret(self._format(value))

        # Fall back to base implementation
        for base in self.bases:
            try:
                return base.format_or_raise(value)
            except NotImplementedError:
                pass

        raise NotImplementedError

    def format(self, value: T) -> Text:
        try:
            return self.format_or_raise(value)
        except NotImplementedError:
            return Text(str(value))

    def parse_or_raise(self, raw: Any) -> T:
        # Own implementation
        if self._parse is not None:
            return self._parse(raw)

        # Fall back to base implementation
        for base in self.bases:
            try:
                return base.parse_or_raise(raw)
            except NotImplementedError:
                pass

        raise NotImplementedError

    def parse(self, raw: Any) -> T:
        try:
            return self.parse_or_raise(raw)
        except NotImplementedError:
            # Fall back to default implementation
            if isinstance(raw, self.value_type):
                return raw
            elif hasattr(self.value_type, "parse"):
                return getattr(self.value_type, "parse")(raw)
            elif hasattr(self.value_type, "interpret"):
                return getattr(self.value_type, "interpret")(raw)
            else:
                return self.value_type(raw)

    def equal_or_raise(self, a: T, b: T) -> bool:
        # Own implementation
        if self._equal is not None:
            return self._equal(a, b)

        # Fall back to base implementation
        for base in self.bases:
            try:
                return base.equal_or_raise(a, b)
            except NotImplementedError:
                pass

        raise NotImplementedError

    def equal(self, a: T, b: T) -> bool:
        try:
            return self.equal_or_raise(a, b)
        except NotImplementedError:
            # Fall back to default implementation
            return a == b

    def check(self, value: T) -> None:
        # Own implementation
        if self._check is not None:
            self._check(value)

        # Check base implementations
        for base in self.bases:
            base.check(value)

        if not isinstance(value, self.value_type):
            raise TypeError(
                f"Expected {self.value_type.__name__}, got {type(value).__name__}."
            )


def decimal_var_type(prec: int | None = None):
    return VarType[float](
        float,
        name=f"decimal(prec={prec})",
        default=lambda: math.nan,
        format=(lambda x: f"{x:.{prec}f}") if prec is not None else None,
        parse=float,
        equal=(lambda a, b: abs(a - b) < 10 ** (-prec)) if prec is not None else None,
    )


# class VarType(Generic[T], ABC):
#     def __init__(self, value_type: type[T], name: str | None = None) -> None:
#         self.value_type = value_type

#         if name is not None:
#             self.name = name
#         elif type(self) is not VarType:
#             self.name = inflection.underscore(
#                 type(self).__name__.removesuffix("Type").removesuffix("Var")
#             )
#         else:
#             self.name = self.value_type.__name__

#     def __repr__(self) -> str:
#         return (
#             f"{type(self).__name__}(name={self.name!r}, value_type={self.value_type!r})"
#         )

#     def __str__(self) -> str:
#         return self.name

#     def format_value(self, value: T) -> Text:
#         return Text(str(value))

#     def parse_value(self, raw: Any, source: str | None = None) -> T:
#         if isinstance(raw, self.value_type):
#             return raw
#         elif hasattr(self.value_type, "parse"):
#             return getattr(self.value_type, "parse")(raw, source=source)
#         elif hasattr(self.type, "interpret"):
#             return getattr(self.value_type, "interpret")(raw)
#         else:
#             return self.value_type(raw)

#     def values_equal(self, a: T, b: T) -> bool:
#         return a == b

#     def check_value(self, value: T) -> None:
#         if not isinstance(value, self.value_type):
#             raise TypeError(
#                 f"Value {value!r} has invalid type for VarType {self}. Expected {self.value_type.__name__}, got {type(value).__name__}."
#             )


# class InheritedVarType(VarType[T]):
#     def __init__(self, base: VarType[T], name: str | None = None) -> None:
#         self.base = base
#         super().__init__(base.value_type, name)

#     def format_value(self, value: T) -> Text:
#         return self.base.format_value(value)

#     def parse_value(self, raw: Any, source: str | None = None) -> T:
#         return self.base.parse_value(raw, source)

#     def values_equal(self, a: T, b: T) -> bool:
#         return self.base.values_equal(a, b)

#     def check_value(self, value: T) -> None:
#         self.base.check_value(value)


# class CombinedVarType(VarType[T]):
#     def __init__(self, types: list[VarType[T]], name: str | None = None) -> None:
#         self.types = types

#         value_type = types[0].value_type
#         for other_type in types[1:]:
#             if value_type is not other_type.value_type:
#                 raise ValueError(
#                     f"Cannot combine VarTypes with different value types: {value_type} and {other_type.value_type}."
#                 )

#         super().__init__(
#             value_type, name or f"combined({', '.join(type.name for type in types)})"
#         )

#     def format_value(self, value: T) -> Text:
#         return self.types[0].format_value(value)

#     def parse_value(self, raw: Any, source: str | None = None) -> T:
#         value = self.types[0].parse_value(raw, source)

#         for type in self.types[1:]:
#             value = type.parse_value(value, source="output_of_other_var_type")

#         return value

#     def values_equal(self, a: T, b: T) -> bool:
#         return self.types[0].values_equal(a, b)

#     def check_value(self, value: T) -> None:
#         for type in self.types:
#             type.check_value(value)


# class DecimalVarType(VarType[float]):
#     def __init__(self) -> None:
#         super().__init__(float, "decimal")

#     def format_value(self, value: float) -> Text:
#         return Text(f"{value:.2f}")

#     def parse_value(self, raw: Any, source: str | None = None) -> float:
#         if isinstance(raw, float):
#             return raw
#         elif isinstance(raw, int):
#             return float(raw)
#         elif isinstance(raw, str):
#             try:
#                 return float(raw)
#             except ValueError:
#                 raise ValueError(f"Invalid decimal value {raw!r} in {source!r}.")
#         else:
#             raise TypeError(
#                 f"Invalid type {type(raw).__name__} for decimal value in {source!r}."
#             )
