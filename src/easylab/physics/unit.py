from typing import Literal, Optional, Union

import sympy
from . import dims, units
from .dim import Dim
from ..lang import Text, TextInput, lang
from ..util import ExprObject


UnitInput = Union["Unit", str, Literal[1], None]


class Unit(ExprObject):
    dim: Dim
    scale: float
    offset: float

    def __init__(
        self,
        label: TextInput,
        dim: Dim = dims.one,
        *,
        scale: float = 1,
        offset: float = 0,
    ) -> None:

        self.dim = dim
        self.scale = float(scale)
        self.offset = float(offset)

        super().__init__(label)

    def __init_from_expr__(self):
        super().__init_from_expr__()

        f = self.create_eval_function()

        unit_deps: list[Unit] = []
        for dep in self.dependencies:
            if isinstance(dep, Unit):
                if dep.offset != 0:
                    raise ValueError("Cannot compose units with offsets.")
                unit_deps.append(dep)
            else:
                raise ValueError(
                    f"Units cannot only be composed of other units. Got {type(dep)}."
                )

        self.dim = f(*(dep.dim for dep in unit_deps))
        self.scale = f(*(dep.scale for dep in unit_deps))
        print([dep.scale for dep in unit_deps])
        print(self.scale)
        self.offset = 0.0  # TODO

        if not isinstance(self.dim, Dim):
            raise ValueError(
                f"Could not create derived unit from expression {self._expr}. Got invalid dim {self.dim}."
            )

        if not isinstance(self.scale, (int, float)):
            raise ValueError(
                f"Could not create derived unit from expression {self._expr}. Got invalid scale {self.scale}."
            )

        self.scale = float(self.scale)
        self.name = None

    @staticmethod
    def parse(input: UnitInput):
        if input is None or input == 1:
            return units.one
        elif isinstance(input, Unit):
            return input
        elif isinstance(input, str):
            expr: sympy.Expr = sympy.parse_expr(input)
            deps = []
            subs = []
            for symb in expr.free_symbols:
                if symb.name in units.all_by_query_strings:
                    unit = units.all_by_query_strings[symb.name]
                    deps.append(unit)
                    print(unit, unit.label.default)
                    subs.append((symb, unit.label.default))

            expr = expr.subs(subs)
            print("expr", expr, repr(expr))

            return Unit.from_expr(expr, deps)

        raise ValueError(f"Cannot parse unit from {input}.")

    def __is_one__(self) -> bool:
        return self.dim.__is_one__() and self.scale == 1 and self.offset == 0

    @property
    def label(self) -> Text:
        return super().label

    @label.setter
    def label(self, label: TextInput):
        super(Unit, type(self)).label.fset(self, lang.mathrm(Text.parse(label)))  # type: ignore

    def prefix(self, prefix_unit: UnitInput):
        prefix_unit = Unit.parse(prefix_unit)
        if self.label is None or prefix_unit.label is None:
            raise ValueError("Cannot prefix units without labels.")

        return Unit(
            prefix_unit.label + self.label,
            dim=prefix_unit.dim * self.dim,
            scale=prefix_unit.scale * self.scale,
        )

    def __mod__(self, other: UnitInput) -> "Unit":
        return Unit.parse(other).prefix(self)

    def is_convertable_to(self, other: UnitInput) -> bool:
        return self.dim == Unit.parse(other).dim

    def convert(self, value: float, to: UnitInput):
        to = Unit.parse(to)
        if not self.is_convertable_to(to):
            raise ValueError(
                f"Cannot convert from unit {self} to unit {to}, since they have different dimensions {self.dim} and {to.dim}, respectively."
            )

        return (value - self.offset) * self.scale / to.scale + to.offset
