from copy import copy
from functools import cache
import math
import re
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    SupportsFloat,
    Union,
    overload,
)
from typing_extensions import Self
from oauthlib import get_debug

import sympy
import numpy as np

from ..util import ExprObject, empty
from ..physics import Unit, units, UnitInput
from ..lang import lang, Text, TextInput
from .var import OutputTarget, Var


ValueInput = Union["Value", float, int, str]

_last_value_index = 0


def get_float_decimals(x: float):
    parts = str(x).split(".", 1)
    if len(parts) == 2 and re.match(r"\d*$", parts[1]):
        return len(parts[1].rstrip("0"))
    return 0


def round_up(x: float, decimals: int):
    return math.ceil(x * 10 ** decimals) / 10 ** decimals


class Value(ExprObject):
    mean: float
    err: float
    prec: int
    unit: Unit

    def __init__(
        self,
        input: ValueInput,
        /,
        *,
        err: Optional[float] = None,
        prec: Optional[int] = None,
        unit: UnitInput = None,
        _expr: Optional[sympy.Expr] = None,
        _label: Optional[TextInput] = None,
        _dependencies: Iterable[ExprObject] = [],
    ) -> None:
        _mean = 0.0
        _err = 0.0
        _prec: Optional[int] = None
        _unit: Optional[Unit] = None

        if isinstance(input, Value):
            _mean = input.mean
            _err = input.err
            _unit = input.unit
            _prec = input.prec
        elif isinstance(input, str):
            s = input.strip()
            if s != "":
                match = re.match(
                    r"(?:(-?[0-9]*)(?:[,.]([0-9]+))?)\s*(?:\(([0-9]+)\))?\s*(.*)", input
                )
                if match is None:
                    raise ValueError

                mean_str, mean_decimals_str, err_int_str, unit_str = match.groups()
                if empty(mean_str):
                    _mean = 0.0
                else:
                    _mean = float(
                        mean_str
                        if mean_decimals_str is None
                        else f"{mean_str}.{mean_decimals_str}"
                    )
                if not empty(unit_str):
                    _unit = units.find(unit_str)

                if not empty(err_int_str):
                    err_int = int(err_int_str)
                    _prec = len(mean_decimals_str or "")
                    _err = float(err_int / 10 ** _prec)

        elif isinstance(input, (int, float)):
            _mean = float(input)

        # Override explicitly specified values
        if err is not None:
            _err = err
        if prec is not None:
            _prec = prec
        if unit is not None:
            unit = Unit.parse(unit)
            if _unit is not None:
                if not unit.is_convertable_to(_unit):
                    raise ValueError(
                        f"Explicitly specified unit ({unit}) is not compatible with unit of input ({_unit}). Input was {input}."
                    )
                _mean = _unit.convert(_mean, unit)
                _err = _unit.convert(_err, unit)
            _unit = unit

        if _prec is None:
            _prec = get_float_decimals(_mean)

        _mean = float(_mean)
        _err = float(_err)

        if type(_prec) != int:
            raise ValueError(f"prec must be an int, got {type(_prec).__name__}.")

        if not isinstance(_unit, Unit):
            raise ValueError(f"unit must be a Unit, got {type(_unit).__name__}.")

        self.mean = _mean
        self.err = _err
        self.prec = _prec
        self.unit = _unit or units.one

        super().__init__(expr=_expr, label=_label, dependencies=_dependencies)

    @classmethod
    def from_expr(cls, expr: sympy.Expr, dependencies: Iterable["ExprObject"]):
        f = cls.create_eval_function(expr, dependencies)

        deps: list[Value] = []
        dep_symbols: list[sympy.Symbol] = []
        dep_err_symbols: list[sympy.Symbol] = []
        for i, dep in enumerate(dependencies):
            if isinstance(dep, Value):
                deps.append(dep)
                dep_symbols.append(sympy.Symbol(f"val{i}"))
                dep_err_symbols.append(sympy.Symbol(f"err{i}"))
            else:
                raise ValueError(
                    f"Values cannot only be composed of other values. Got {type(dep)}."
                )

        mean = float(f(*(dep.mean for dep in deps)))

        expr = f(*dep_symbols)
        err_expr = sympy.sqrt(
            sum(
                (sympy.diff(expr, symb) * err_symb) ** 2
                for symb, err_symb in zip(dep_symbols, dep_err_symbols)
            )
        )
        # print(err_expr)
        err_func = sympy.lambdify(dep_symbols + dep_err_symbols, err_expr)
        err = float(err_func(*(dep.mean for dep in deps), *(dep.err for dep in deps)))

        prec = max(dep.prec for dep in deps)  # TODO: Does this make sense?
        unit = f(*(dep.unit for dep in deps))

        return Value(
            mean, err=err, prec=prec, unit=unit, _expr=expr, _dependencies=dependencies
        )

    # def __init_from_expr__(self):
    #     f = self.create_eval_function()

    #     deps: list[Value] = []
    #     dep_symbols: list[sympy.Symbol] = []
    #     dep_err_symbols: list[sympy.Symbol] = []
    #     for i, dep in enumerate(self.dependencies):
    #         if isinstance(dep, Value):
    #             deps.append(dep)
    #             dep_symbols.append(sympy.Symbol(f"val{i}"))
    #             dep_err_symbols.append(sympy.Symbol(f"err{i}"))
    #         else:
    #             raise ValueError(
    #                 f"Values cannot only be composed of other values. Got {type(dep)}."
    #             )

    #     self.mean = float(f(*(dep.mean for dep in deps)))

    #     expr = f(*dep_symbols)
    #     err_expr = sympy.sqrt(
    #         sum(
    #             (sympy.diff(expr, symb) * err_symb) ** 2
    #             for symb, err_symb in zip(dep_symbols, dep_err_symbols)
    #         )
    #     )
    #     # print(err_expr)
    #     err_func = sympy.lambdify(dep_symbols + dep_err_symbols, err_expr)
    #     self.err = float(
    #         err_func(*(dep.mean for dep in deps), *(dep.err for dep in deps))
    #     )

    #     self.prec = max(dep.prec for dep in deps)  # TODO: Does this make sense?
    #     self.unit = f(*(dep.unit for dep in deps))

    #     # dep_errs = [dep.err_value for dep in deps]

    #     # self.mean = f(*(dep.mean for dep in deps))

    #     # uncertainty_expr = sympy.sqrt(
    #     #     sum(
    #     #         (self.derivative_expr(dep) * uncertainty) ** 2
    #     #         for dep, uncertainty in zip(deps, dep_errs)
    #     #     )
    #     # )
    #     # print(uncertainty_expr)
    #     # uncertainty_func = sympy.lambdify(
    #     #     [dep.expr for dep in (deps + dep_errs)], uncertainty_expr
    #     # )
    #     # self.err = uncertainty_func(*(dep.mean for dep in (deps + dep_errs)))
    #     # self.precision = None
    #     # self.unit = f(*(dep.unit for dep in deps))

    def derivative_expr(self, dep: "Value"):
        return self.expr_or_fail().diff(dep.expr)

    @property
    @cache
    def err_value(self):
        return Value(
            self.err,
            prec=self.prec,
            unit=self.unit,
        )

    @property
    def has_err(self):
        return self.err != 0 and self.err != (0, 0)

    @property
    @cache
    def rounded_err(self):
        if isinstance(self.err, tuple):
            return (round_up(self.err[0], self.prec), round_up(self.err[1], self.prec))
        else:
            return round_up(self.err, self.prec)

    @property
    def text(self):
        text = lang.number(self.mean, self.prec)

        if self.has_err:
            rounded_err = self.rounded_err
            if isinstance(rounded_err, tuple):
                text += lang.substack(
                    "+" + lang.number(rounded_err[0], self.prec),
                    "-" + lang.number(rounded_err[1], self.prec),
                )
            else:
                text += " " + lang.pm + " " + lang.number(rounded_err, self.prec)
        if self.unit != units.one:
            if self.has_err:
                text = lang.par(text)
            text += lang.small_space + self.unit.text
        return text

    def __str__(self) -> str:
        return self.text.default

    def __repr__(self) -> str:
        return f"Value({self.mean}, err={self.err}, prec={self.prec}, unit={self.unit})"

    def __float__(self) -> float:
        return self.mean

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Value):
            other = Value(other)

        return (
            other.mean == self.mean
            and other.err == self.err
            and other.unit == self.unit
            and other.prec == self.prec
        )

    def __hash__(self) -> int:
        return hash((self.mean, self.err, self.unit, self.prec))

    def __le__(self, other: "Value") -> bool:
        return self.mean <= other.mean

    def __lt__(self, other: "Value") -> bool:
        return self.mean < other.mean

    def __ge__(self, other: "Value") -> bool:
        return self.mean >= other.mean

    def __gt__(self, other: "Value") -> bool:
        return self.mean > other.mean

    def convert_to(self, unit: Unit):
        if self.unit == unit:
            return self
        return Value(self, unit=unit)

    def copy(
        self,
        *,
        mean: Optional[float] = None,
        err: Optional[float] = None,
        prec: Optional[int] = None,
        unit: Optional[Unit] = None,
    ) -> "Value":
        return Value(
            mean or self.mean,
            err=err or self.err,
            prec=prec or self.prec,
            unit=unit or self.unit,
        )

    def remove_unit(self):
        return self.copy(unit=units.one)

    def remove_err(self):
        return self.copy(err=0)


class ValueVar(Var[Value]):
    unit: Unit
    prec: Optional[int]
    fallback_err: Optional[float]

    def __init__(
        self,
        label: Optional[TextInput] = None,
        *,
        unit: UnitInput = None,
        prec: Optional[int] = None,
        default: Optional[ValueInput] = None,
        format: Optional[Callable[[Value], TextInput]] = None,
        parse: Optional[Callable[[Any], Value]] = None,
        check: Optional[Callable[[Value], Union[bool, str, None]]] = None,
        fallback_err: Optional[float] = None,
        name: Optional[str] = None,
        auto_name: bool = True,
    ):
        super().__init__(
            label,
            type=Value,
            format=format,
            parse=parse,
            check=check,
            name=name,
            auto_name=auto_name,
        )
        self.unit = Unit.parse(unit)
        self.prec = prec
        self.fallback_err = fallback_err

        if default is not None:
            self.default = Value(default, unit=self.unit, prec=prec)

    def __init_from_expr__(self):
        super().__init_from_expr__()

        f = self.create_eval_function()

        var_deps: list[Var] = [d for d in self.dependencies if isinstance(d, Var)]

        self.unit = f(
            *(v.unit if isinstance(v, ValueVar) else units.one for v in var_deps)
        )
        self.prec = None
        self.fallback_err = None

    def _check(self, value: Value):
        if not value.unit.is_convertable_to(self.unit):
            raise ValueError(
                f"Value {value} has invalid unit {value.unit} for variable {self}. Expected a unit that is convertable to {self.unit}. Dimensions are {value.unit.dim} and {self.unit.dim}."
            )

    def _parse_fallback(self, input: Any) -> Value:
        value = Value(input, prec=self.prec, unit=self.unit)
        self._check(value)  # Check value before trying to convert unit

        # var = copy(value.var)
        # var.prec = self.prec
        # value.var = var
        return value.convert_to(self.unit)

    def _output_fallback(self, value: Value, target: OutputTarget):
        if target == "plot":
            return value.mean
        elif target == "plot_err":
            return value.err
        return super()._output_fallback(value, target)

    @property
    @cache
    def err(self) -> "ValueVar":
        if self.is_computed:
            value_var_deps = [
                dep for dep in self.dependencies if isinstance(dep, ValueVar)
            ]
            dep_err_vars = [dep.err for dep in value_var_deps]

            return ValueVar.from_expr(
                expr=sympy.sqrt(
                    sum(
                        (self.expr.diff(dep.expr) * dep_err.expr) ** 2
                        for dep, dep_err in zip(value_var_deps, dep_err_vars)
                    )
                ),
                dependencies=[*value_var_deps, *dep_err_vars],
            )
        else:
            return ValueVar(
                lang.Delta
                + (self.label if self.label is not None else lang.par(self.text)),
                unit=self.unit,
                prec=self.prec,
                default=0,
                format=self._format_func,
                parse=self._parse_func,
                check=self._check_func,
                fallback_err=0,
                auto_name=False,
            )

    def copy(
        self,
        *,
        label: Optional[TextInput] = None,
        unit: Optional[Unit] = None,
        prec: Optional[int] = None,
        default: Optional[ValueInput] = None,
        format: Optional[Callable[[Value], TextInput]] = None,
        parse: Optional[Callable[[Any], Value]] = None,
        check: Optional[Callable[[Value], Union[bool, str, None]]] = None,
        fallback_err: Optional[float] = None,
        name: Optional[str] = None,
    ):
        return ValueVar(
            label=label or self.label,
            unit=unit or self.unit,
            prec=prec or self.prec,
            default=default or self.default,
            format=format or self._format_func,
            parse=parse or self._parse_func,
            check=check or self._check_func,
            fallback_err=fallback_err or self.fallback_err,
            name=name or self.name,
        )

    def remove_unit(self) -> Self:
        default = self.default.remove_unit() if self.default is not None else None

        return self.copy(
            label=self.label + lang.space + "/" + lang.space + self.unit.label,
            unit=units.one,
            default=default,
            name=self.name + "_nounit",
        )

    def remove_err(self) -> Self:
        return self.copy(
            fallback_err=None,
            name=self.name + "_noerr",
        )
