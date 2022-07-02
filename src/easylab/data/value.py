from copy import copy
from functools import cache
from typing import Any, Callable, Optional, Union, overload
from typing_extensions import Self

import sympy

from ..util import ExprObject
from ..physics import Unit, units
from ..lang import lang, Text, TextInput
from .var import OutputTarget, Var

ValueInput = Union["Value", float, int]

_last_value_index = 0


class Value(ExprObject):
    mean: float
    err: float
    var: "ValueVar"

    @overload
    def __init__(
        self,
        mean: float,
        *,
        err: Optional[float] = None,
        var: Optional["ValueVar"] = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        mean: float,
        *,
        err: Optional[float] = None,
        precision: Optional[int] = None,
        unit: Unit = units.one,
    ) -> None:
        ...

    def __init__(
        self,
        mean: float,
        *,
        err: Optional[float] = None,
        precision: Optional[int] = None,
        unit: Optional[Unit] = None,
        var: Optional["ValueVar"] = None,
    ) -> None:
        self.mean = float(mean)
        self.err = err or (0 if var is None else var.fallback_err or 0)

        if var is None:
            global _last_value_index
            var = ValueVar(
                Text(
                    "__val" + str(_last_value_index),
                    latex="\\mathrm{val}_{" + str(_last_value_index) + "}",
                ),
                unit=unit or units.one,
                prec=precision,
                auto_name=False,
            )
            _last_value_index += 1
        elif precision is not None or unit is not None:
            raise ValueError(
                "Cannot specify precision or unit for a value with a specific var."
            )

        self.var = var

        super().__init__(var.expr, [])

    def __init_from_expr__(self):
        f = self.create_eval_function()

        deps: list[Value] = []
        for dep in self.dependencies:
            if isinstance(dep, Value):
                deps.append(dep)
            else:
                raise ValueError(
                    f"Values cannot only be composed of other values. Got {type(dep)}."
                )

        self.mean = f(*(dep.mean for dep in deps))
        self.err = 0  # TODO
        self.var = f(*(dep.var for dep in deps))

        # dep_errs = [dep.err_value for dep in deps]

        # self.mean = f(*(dep.mean for dep in deps))

        # uncertainty_expr = sympy.sqrt(
        #     sum(
        #         (self.derivative_expr(dep) * uncertainty) ** 2
        #         for dep, uncertainty in zip(deps, dep_errs)
        #     )
        # )
        # print(uncertainty_expr)
        # uncertainty_func = sympy.lambdify(
        #     [dep.expr for dep in (deps + dep_errs)], uncertainty_expr
        # )
        # self.err = uncertainty_func(*(dep.mean for dep in (deps + dep_errs)))
        # self.precision = None
        # self.unit = f(*(dep.unit for dep in deps))

    def derivative_expr(self, dep: "Value"):
        return self.expr.diff(dep.expr)

    @property
    def precision(self):
        return self.var.prec

    @property
    def unit(self):
        return self.var.unit

    @property
    @cache
    def err_value(self):
        return Value(
            self.err,
            var=self.var.err,
        )

    @property
    def has_err(self):
        return abs(self.err) > 1e-10

    @staticmethod
    def parse(
        input: ValueInput,
        *,
        var_hint: Optional["ValueVar"] = None,
        fresh: bool = False,
    ) -> "Value":
        if input is None:
            if var_hint is not None and var_hint.default is not None:
                return var_hint.default
            else:
                return Value(0, var=var_hint)
        elif isinstance(input, Value):
            return copy(input) if fresh else input
        elif isinstance(input, (float, int)):
            return Value(input, var=var_hint)
        else:
            raise ValueError(f"Cannot parse {input} as Value.")

    @property
    def value_text(self):
        text = lang.number(self.mean, self.precision)

        if self.has_err:
            if isinstance(self.err, tuple):
                text += lang.substack(
                    "+" + lang.number(self.err[0], self.precision),
                    "-" + lang.number(self.err[1], self.precision),
                )
            else:
                text += " " + lang.pm + " " + lang.number(self.err, self.precision)
        if self.unit != units.one:
            if self.has_err:
                text = lang.par(text)
            text += lang.small_space + self.unit.text
        return text

    @property
    def text(self):
        return self.value_text

    @property
    def eq_text(self):
        return self.var.label + " = " + self.value_text

    def __str__(self) -> str:
        return self.text.default

    def __repr__(self) -> str:
        return f"Value({self.mean}, err={self.err}, var={self.var!r})"

    def __float__(self) -> float:
        return self.mean

    def __eq__(self, other: Any) -> bool:
        value = Value.parse(other, var_hint=self.var)
        return (
            value.mean == self.mean
            and value.err == self.err
            and value.unit == self.unit
            and value.precision == self.precision
        )

    def __hash__(self) -> int:
        return hash((self.mean, self.err, self.unit, self.precision))

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
        return Value(
            self.unit.convert(self.mean, unit),
            err=self.unit.convert(self.err, unit),
            # uncertainty=(
            #     self.unit.convert(self.uncertainty[0], unit),
            #     self.unit.convert(self.uncertainty[1], unit),
            # )
            # if isinstance(self.uncertainty, tuple)
            # else self.unit.convert(self.uncertainty, unit),
            precision=self.precision,
            unit=unit,
        )

    def remove_unit(self):
        result = copy(self)
        result.var = self.var.remove_unit()
        return result

    def remove_err(self):
        result = copy(self)
        result.err = 0
        return result


class ValueVar(Var[Value]):
    unit: Unit
    prec: Optional[int]
    fallback_err: Optional[float]

    def __init__(
        self,
        label: Optional[TextInput] = None,
        *,
        unit: Unit = units.one,
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
        self.unit = unit
        self.prec = prec
        self.fallback_err = fallback_err

        if default is not None:
            self.default = Value.parse(default, var_hint=self)

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
        if value.unit.dim != self.unit.dim:
            raise ValueError(
                f"Value {value} has invalid unit {value.unit} for variable {self}. Expected a unit that is convertable to {self.unit}."
            )

    def _parse_fallback(self, input: Any) -> Value:
        value = Value.parse(input, var_hint=self, fresh=True)
        value.var = self
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

    def remove_unit(self) -> Self:
        default = copy(self.default)
        if default is not None:
            default_var = copy(default.var)
            default_var.unit = units.one
            default.var = default_var

        return ValueVar(
            self.label + lang.space + "/" + lang.space + self.unit.label,
            unit=units.one,
            prec=self.prec,
            default=default,
            format=self._format_func,
            parse=self._parse_func,
            check=self._check_func,
            fallback_err=self.fallback_err,
            name=self.name + "_nounit",
        )

    def remove_err(self) -> Self:
        return ValueVar(
            self.label,
            unit=units.one,
            prec=self.prec,
            default=self.default,
            format=self._format_func,
            parse=self._parse_func,
            check=self._check_func,
            fallback_err=None,
            name=self.name + "_noerr",
        )

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
