from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Literal, Optional, TypeVar, Union, cast
from typing_extensions import TypeGuard

from ..util import Comparable
from ..lang import Text, lang, text, HasText
from . import var as m_var

_T = TypeVar("_T")
_C = TypeVar("_C", bound=Comparable)


ConstraintInput = Union[
    "Constraint[_T]",
    tuple["m_var.Var[_T]", _T],  # Equal, e.g. (x, 1)
    tuple["m_var.Var[_T]", _T, _T],  # Bounds, e.g. (x, 0, 1)
    tuple[
        "m_var.Var[_T]", Literal[">=", "<=", ">", "<"], _T
    ],  # Compare, e.g. (x, ">", 0)
]


def is_constraint_input(input: Any) -> TypeGuard[ConstraintInput]:
    return isinstance(input, Constraint) or (
        isinstance(input, tuple)
        and len(input) in (2, 3)
        and isinstance(input[0], m_var.Var)
    )


class Constraint(Generic[_T], HasText):
    var: "m_var.Var[_T]"

    def __init__(self, var: "m_var.Var[_T]"):
        self.var = var

    @staticmethod
    def parse(input: ConstraintInput[_T]) -> "Constraint[_T]":
        if isinstance(input, Constraint):
            return input

        elif isinstance(input, tuple) and isinstance(input[0], m_var.Var):
            var, *args = input
            if len(args) == 1:
                return EqualConstraint(var, cast(_T, args[0]))
            elif len(args) == 2 and isinstance(args[0], str):
                if args[0] == "<=":
                    return BoundsConstraint(var, None, args[1])  # type: ignore
                elif args[0] == ">=":
                    return BoundsConstraint(var, args[1], None)  # type: ignore
                elif args[0] == "<":
                    return BoundsConstraint(var, None, args[1], include_max=False)  # type: ignore
                elif args[0] == ">":
                    return BoundsConstraint(var, args[1], None, include_min=False)  # type: ignore
                else:
                    raise ValueError(f"Invalid comparison operator {args[0]}.")
            elif (
                len(args) == 2
                and isinstance(args[0], Comparable)
                and isinstance(args[1], Comparable)
            ):
                return BoundsConstraint(var, args[0], args[1])  # type: ignore

        raise ValueError(f"Cannot patse VarConstraint from {input}.")

    @abstractmethod
    def includes(self, other: "Constraint[_T]") -> bool:
        pass

    @property
    @abstractmethod
    def text(self) -> Text:
        pass

    @property
    def value(self) -> Optional[_T]:
        pass

    def __and__(self, other: "Constraint[_T]") -> "Constraint[_T]":
        return AndConstraint(self, other)

    def __or__(self, other: "Constraint[_T]") -> "Constraint[_T]":
        return OrConstraint(self, other)


class AnyConstraint(Constraint[_T]):
    def includes(self, other: "Constraint[_T]") -> bool:
        if isinstance(other, AnyConstraint) and other.var == self.var:
            return True
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AnyConstraint) and self.var == other.var

    def __hash__(self) -> int:
        return hash(self.var)

    @property
    def text(self) -> Text:
        return Text(f"any {self.var}", repr=f"AnyConstraint({self.var})")


class EqualConstraint(Constraint[_T]):
    _value: _T

    def __init__(self, var: "m_var.Var[_T]", value: Any):
        super().__init__(var)

        self._value = var.parse(value)

    @property
    def value(self) -> _T:
        return self._value

    def includes(self, other: Constraint[_T]) -> bool:
        return (
            isinstance(other, EqualConstraint)
            and other.var == self.var
            and other._value == self._value
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, EqualConstraint)
            and self.var == other.var
            and self._value == other._value
        )

    def __hash__(self) -> int:
        return hash((self.var, self._value))

    @property
    def text(self) -> Text:
        return Text(
            f"{self.var} = {self._value}",
            repr=f"EqualConstraint({self.var}, {self._value})",
        )


class BoundsConstraint(Constraint[_C]):
    min: Optional[_C]
    max: Optional[_C]
    include_min: bool = True
    include_max: bool = True

    def __init__(
        self,
        var: "m_var.Var[_C]",
        min: Any = None,
        max: Any = None,
        *,
        include_min: bool = True,
        include_max: bool = True,
    ):
        super().__init__(var)

        min_value = var.parse(min) if min is not None else None
        max_value = var.parse(max) if max is not None else None

        if (min_value is not None and not isinstance(min_value, Comparable)) or (
            max_value is not None and not isinstance(max_value, Comparable)
        ):
            raise ValueError(
                f"Parameters min and max must be comparable (using < and >), got {type(min_value)} and {type(max_value)}."
            )

        self.min = min_value
        self.max = max_value
        self.include_min = include_min
        self.include_max = include_max

    @property
    def value(self) -> Optional[_C]:
        if self.min is not None and self.max is not None and self.min == self.max:
            return self.min

    def fulfills_min(self, value: _C):
        if self.min is None:
            return True
        if self.include_min:
            return value >= self.min
        else:
            return value > self.min

    def fulfills_max(self, value: _C):
        if self.max is None:
            return True
        if self.include_max:
            return value <= self.max
        else:
            return value < self.max

    def fulfills_bounds(self, value: _C):
        return self.fulfills_min(value) and self.fulfills_max(value)

    def includes(self, other: Constraint[_C]) -> bool:
        if other.var != self.var:
            return False

        if isinstance(other, EqualConstraint):
            self.var.check(other.value)

            return self.fulfills_bounds(other.value)
        elif isinstance(other, BoundsConstraint):
            if self.min is not None and other.min is not None:
                self.var.check(other.min)
                return self.fulfills_min(other.min)
            if self.max is not None and other.max is not None:
                self.var.check(other.max)
                return self.fulfills_max(other.max)
            return True
        else:
            return False

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BoundsConstraint)
            and self.var == other.var
            and self.min == other.min
            and self.max == other.max
            and self.include_min == other.include_min
            and self.include_max == other.include_max
        )

    def __hash__(self) -> int:
        return hash((self.var, self.min, self.max, self.include_min, self.include_max))

    @property
    def text(self) -> Text:
        if self.min is None and self.max is None:
            t = "any" + lang.space + self.var.text
        elif self.min is None:
            symbol = lang.leq if self.include_max else lang.lt
            t = self.var.text + symbol + text(self.max)
        elif self.max is None:
            symbol = lang.geq if self.include_min else lang.gt
            t = self.var.text + symbol + text(self.min)
        else:
            symbol1 = lang.leq if self.include_min else lang.lt
            symbol2 = lang.leq if self.include_max else lang.lt
            t = text(self.min) + symbol1 + self.var.text + symbol2 + text(self.max)

        return t.extend(
            repr=f"BoundsConstraint({self.var}, {self.min}, {self.max}, {self.include_min}, {self.include_max})"
        )


class AndConstraint(Constraint[_T]):
    constraints: tuple[Constraint[_T], ...]

    def __init__(self, *constraints: Constraint[_T]):
        if len(constraints) == 0:
            raise ValueError("Must have at least one constraint.")

        self.constraints = constraints

        var = self.constraints[0].var
        for constraint in self.constraints[1:]:
            if constraint.var != var:
                raise ValueError(
                    f"All constraints must have the same var, got {var} and {constraint.var}."
                )

        super().__init__(var)

    @property
    def value(self) -> Optional[_T]:
        value = self.constraints[0].value
        for constraint in self.constraints[1:]:
            v = constraint.value
            if v != value:
                return None

        return value

    def includes(self, other: Constraint[_T]) -> bool:
        return all(c.includes(other) for c in self.constraints)

    @property
    def text(self) -> Text:
        return (
            Text(" and ", latex=" \\wedge ")
            .join(c.text for c in self.constraints)
            .extend(repr=f"AndConstraint({self.var}, {self.constraints})")
        )


class OrConstraint(Constraint[_T]):
    constraints: tuple[Constraint[_T], ...]

    def __init__(self, *constraints: Constraint[_T]):
        if len(constraints) == 0:
            raise ValueError("Must have at least one constraint.")

        self.constraints = constraints

        var = self.constraints[0].var
        for constraint in self.constraints[1:]:
            if constraint.var != var:
                raise ValueError(
                    f"All constraints must have the same var, got {var} and {constraint.var}."
                )

        super().__init__(var)

    @property
    def value(self) -> Optional[_T]:
        value = self.constraints[0].value
        for constraint in self.constraints[1:]:
            v = constraint.value
            if v != value:
                return None

        return value

    def includes(self, other: Constraint[_T]) -> bool:
        return any(c.includes(other) for c in self.constraints)

    @property
    def text(self) -> Text:
        return (
            Text(" or ", latex=" \\vee ")
            .join(c.text for c in self.constraints)
            .extend(repr=f"OrConstraint({self.var}, {self.constraints})")
        )


class CustomConstraint(Constraint[_T]):
    _includes: Callable[[Constraint[_T]], bool]

    def __init__(
        self, var: "m_var.Var[_T]", includes: Callable[[Constraint[_T]], bool]
    ):
        self._includes = includes
        super().__init__(var)

    def includes(self, other: Constraint[_T]) -> bool:
        return self._includes(other)

    @property
    def text(self) -> Text:
        return Text(f"CustomConstraint({self.var}, {self._includes.__name__})")
