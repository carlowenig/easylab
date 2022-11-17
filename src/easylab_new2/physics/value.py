from __future__ import annotations
from abc import ABC, abstractmethod
import functools
import math
import re
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    SupportsFloat,
    SupportsInt,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import TypeGuard

from ..data import VarType, Var
from ..lang import Text, lang
from ..internal_util import format_args

Unit = Any  # TODO


def is_unit(x):
    return False  # TODO


def get_float_prec(float_str: str) -> int:
    pattern = r"(?:[-+]?(?:[0-9]*[.,]?(?P<digits>[0-9]+))|(?:[0-9]+[.,]))([eE](?P<exp>[-+]?[0-9]+))?"
    match = re.match(pattern, float_str)
    if match is None:
        raise ValueError(f"Invalid float string: {float_str}")
    digits = len(match.group("digits") or "")
    exp = int(match.group("exp") or 0)
    return digits - exp


def infer_prec_from_unc(unc: float | None, significant_digits: int = 2) -> int | None:
    if unc is None or unc == 0:
        return None
    # if unc >= 1:
    #     return 0
    return math.ceil(-math.log10(unc) + significant_digits - 1)


def infer_prec_from_val(val: float) -> int:
    s = str(val)
    if "." in s:
        return len(s) - s.index(".") - 1
    else:
        return 0


def infer_prec(val: float, unc: float | None, significant_digits: int = 2) -> int:
    result = infer_prec_from_unc(unc, significant_digits)
    if result is None:
        result = infer_prec_from_val(val)
    return result


PhysicalValueLike = Union[
    "PhysicalValue",
    str,
    SupportsFloat,
]


def is_physical_value_like(value: Any) -> TypeGuard[PhysicalValueLike]:
    return isinstance(value, (PhysicalValue, str, SupportsFloat))


class PhysicalValue(ABC):
    @staticmethod
    def interpret(input: PhysicalValueLike) -> PhysicalValue:
        if isinstance(input, PhysicalValue):
            return input
        elif isinstance(input, str):
            num_pattern = (
                r"(?:[-+]?(?:[0-9]*[.,]?[0-9]+)|(?:[0-9]+[.,]))([eE][-+]?[0-9]+)?"
            )
            val_pattern = r"(?P<val>" + num_pattern + ")"
            unc_int_pattern = r"(?:\(\s*(?P<unc_int>[0-9]+)\s*\))"
            pm_pattern = r"\+\/?-?"
            unc_val_pattern = (
                r"(?:" + pm_pattern + r"\s*(?P<unc_val>" + num_pattern + r"))"
            )
            unc_pattern = r"(?:" + unc_int_pattern + "|" + unc_val_pattern + r")?"
            exp_pattern = r"([eE](?P<exp>[-+]?[0-9]+))?"
            unit_pattern = r"(?P<unit>.*)"

            pattern = (
                r"^\s*"
                + val_pattern
                + r"\s*"
                + unc_pattern
                + r"\s*"
                + exp_pattern
                + r"\s*"
                + unit_pattern
                + r"$"
            )
            # print("pattern", pattern)

            match = re.match(pattern, input)

            if match is None:
                raise ValueError("Invalid input string: " + input)

            val = match.group("val").replace(",", ".")
            prec = get_float_prec(val)
            val = float(val)
            unc_int = match.group("unc_int")
            unc_val = match.group("unc_val")
            exp = match.group("exp")
            unit = match.group("unit")

            if unc_int is not None:
                unc = 10 ** -prec * int(unc_int)
            elif unc_val is not None:
                unc_val = unc_val.replace(",", ".")
                prec = max(prec, get_float_prec(unc_val))
                unc = float(unc_val.replace(",", "."))
            else:
                unc = 0

            if exp is not None:
                exp = int(exp)
                val *= 10 ** exp
                unc *= 10 ** exp
                prec -= exp

            return MeasuredValue(val, unc, unit, prec)

        try:
            return MeasuredValue(float(input))
        except Exception as e:
            raise TypeError(
                f"Cannot interpret type {type(input)} as a measured value. {e}"
            )

    val: float
    unc: float | None
    unit: Unit
    prec: int

    @property
    def rounded_val(self):
        prec = self.prec
        return round(self.val * 10 ** prec) / 10.0 ** prec

    @property
    def rounded_unc(self):
        unc = self.unc
        if unc is None:
            return None

        prec = self.prec
        return math.ceil(unc * 10 ** prec) / 10.0 ** prec

    def format_num(self, num: float) -> Text:
        prec = self.prec
        if prec < -3:
            exp = math.floor(math.log10(num))
            mantissa = num / 10 ** exp
            mantissa_prec = prec + exp
            return (
                f"{mantissa:.{mantissa_prec}f}"
                + " "
                + lang.cdot
                + " "
                + Text("10").superscript(exp)
            )
        else:
            return Text(f"{num:.{max(0, prec)}f}")

    @property
    def text(self):
        t = self.format_num(self.rounded_val)

        unc = self.rounded_unc
        if unc is not None:
            t += " " + lang.pm + " " + self.format_num(unc)

        unit = self.unit
        if unit is not None:  # TODO: check for units.one
            if unc is not None:
                t = lang.par(t)
            t += Text(unit)

        return t

    def __str__(self):
        return self.text.ascii

    def __repr__(self):
        return self.text.ascii

    # def __eq__(self, other):
    #     return (
    #         isinstance(other, PhysicalValue)
    #         and self.val == other.val
    #         and self.unc == other.unc
    #         and self.unit == other.unit
    #         and self.prec == other.prec
    #     )

    # def __hash__(self):
    #     return hash((self.val, self.unc, self.unit, self.prec))

    def get_dependencies(self) -> Iterable[MeasuredValue]:
        return []

    def __add__(self, other: PhysicalValueLike):
        return SymbolicallyComputedValue("a + b", a=self, b=other)

    def __radd__(self, other: PhysicalValueLike):
        return SymbolicallyComputedValue("a + b", a=other, b=self)

    def __sub__(self, other: PhysicalValueLike):
        return SymbolicallyComputedValue("a - b", a=self, b=other)

    def __rsub__(self, other: PhysicalValueLike):
        return SymbolicallyComputedValue("a - b", a=other, b=self)

    def __mul__(self, other: PhysicalValueLike):
        return SymbolicallyComputedValue("a * b", a=self, b=other)

    def __rmul__(self, other: PhysicalValueLike):
        return SymbolicallyComputedValue("a * b", a=other, b=self)

    def __truediv__(self, other: PhysicalValueLike):
        return SymbolicallyComputedValue("a / b", a=self, b=other)

    def __rtruediv__(self, other: PhysicalValueLike):
        return SymbolicallyComputedValue("a / b", a=other, b=self)

    def __pow__(self, other: PhysicalValueLike):
        return SymbolicallyComputedValue("a ** b", a=self, b=other)

    def __rpow__(self, other: PhysicalValueLike):
        return SymbolicallyComputedValue("a ** b", a=other, b=self)

    def __neg__(self):
        return SymbolicallyComputedValue("-a", a=self)

    def __pos__(self):
        return self

    def __abs__(self):
        return SymbolicallyComputedValue("abs(a)", a=self)


class MeasuredValue(PhysicalValue):
    def __init__(
        self,
        val: SupportsFloat,
        unc: SupportsFloat | None = None,
        unit: Any = None,
        prec: SupportsInt | None = None,
    ):
        self.val = float(val)
        self.unc = float(unc) if unc is not None else None
        self.unit = unit  # TODO: Unit.interpret(unit)
        self.prec = int(prec) if prec is not None else infer_prec(self.val, self.unc)


class ComputedValue(PhysicalValue):
    def __init__(
        self,
        params: Iterable[PhysicalValueLike],
    ):
        self.params = [PhysicalValue.interpret(p) for p in params]

    @abstractmethod
    def _compute_val(self, *param_vals: float) -> float:
        ...

    @abstractmethod
    def _compute_derivative(self, param_index: int) -> float:
        ...

    @property
    @functools.cache
    def val(self) -> float:
        return self._compute_val(*(dep.val for dep in self.params))

    @property
    @functools.cache
    def unc(self) -> float | None:
        sum_ = 0

        uncertainties = []

        for param in self.params:
            unc = param.unc
            if unc is None:
                return None
            uncertainties.append(unc)

        for i, unc in enumerate(uncertainties):
            sum_ += (self._compute_derivative(i) * unc) ** 2

        return math.sqrt(sum_)

    @property
    @functools.cache
    def unit(self) -> Unit:
        return None  # TODO

    @property
    @functools.cache
    def prec(self) -> int:
        return infer_prec(self.val, self.unc)

    def get_dependencies(self) -> Iterable[MeasuredValue]:
        for param in self.params:
            if isinstance(param, MeasuredValue):
                yield param
            else:
                yield from param.get_dependencies()


T = TypeVar("T")


class Relative(Generic[T]):
    def __init__(self, value: T):
        self.value = value

    def eval(self, base: T) -> T:
        return self.value * base  # type: ignore

    def __repr__(self) -> str:
        return f"Relative({self.value!r})"

    def __str__(self) -> str:
        return f"rel({self.value})"


def rel(value: T) -> Relative[T]:
    return Relative(value)


AbsoluteOrRelative = Union[T, Relative[T]]


def eval_rel(value: AbsoluteOrRelative[T], base: T) -> T:
    if isinstance(value, Relative):
        return value.eval(base)
    else:
        return value


class NumericallyComputedValue(ComputedValue):
    def __init__(
        self,
        params: Iterable[PhysicalValueLike],
        func: Callable[..., Any],
        derivative_dx: AbsoluteOrRelative[float] = rel(1e-6),
    ):
        self.func = func
        self.derivative_dx = derivative_dx
        super().__init__(params)

    def _compute_val(self, *param_vals: float) -> float:
        return float(self.func(*param_vals))

    def _compute_derivative(self, param_index: int) -> float:
        param_vals = [p.val for p in self.params]

        func = lambda param_val: self._compute_val(
            *param_vals[:param_index], param_val, *param_vals[param_index + 1 :]
        )

        from scipy.misc import derivative

        return derivative(
            func,
            param_vals[param_index],
            dx=eval_rel(self.derivative_dx, param_vals[param_index]),
        )


class SymbolicallyComputedValue(ComputedValue):
    @staticmethod
    def from_function(
        params: Iterable[PhysicalValue],
        func: Callable[..., Any],
    ):
        from sympy import Symbol, Expr

        params_dict = {f"p_{i}": p for i, p in enumerate(params)}

        symbols = [Symbol(name) for name in params_dict]

        expr: Expr = func(*symbols)

        if not isinstance(expr, Expr):
            raise TypeError(
                f"Symbolically computed value must return sympy expression. Got {type(expr)}."
            )

        return SymbolicallyComputedValue(expr, **params_dict)

    def __init__(
        self,
        expr: Any,
        **params_by_name: PhysicalValueLike,
    ):
        from sympy import Expr, sympify, Symbol

        self.param_symbols = {name: Symbol(name) for name in params_by_name}

        self.params_by_symbol = {
            self.param_symbols[name]: PhysicalValue.interpret(p)
            for name, p in params_by_name.items()
        }
        self.expr: Expr = sympify(expr, locals=self.param_symbols)

        self.derivatives: list[Expr] = [
            self.expr.diff(symbol) for symbol in self.params_by_symbol
        ]

        super().__init__(self.params_by_symbol.values())

    def _compute_val(self, *param_vals: float) -> float:
        val = self.expr.evalf(
            subs={symbol: val for symbol, val in zip(self.params_by_symbol, param_vals)}  # type: ignore
        )

        try:
            return float(val)  # type: ignore
        except TypeError as e:
            raise TypeError(f"Could not convert computed val {val} to float. {e}")

    def _compute_derivative(self, param_index: int) -> float:
        derivative = self.derivatives[param_index]

        result = derivative.evalf(
            subs={symbol: param.val for symbol, param in self.params_by_symbol.items()}  # type: ignore
        )

        try:
            return float(result)  # type: ignore
        except TypeError as e:
            raise TypeError(
                f"Could not convert computed derivative {result} to float. {e}"
            )


@overload
def value(measured: PhysicalValueLike, /) -> MeasuredValue:
    ...


@overload
def value(expr: str, **params: PhysicalValue) -> SymbolicallyComputedValue:
    ...


@overload
def value(
    val: SupportsFloat,
    unc: SupportsFloat | None = None,
    unit: Unit | None = None,
    prec: SupportsInt | None = None,
    /,
) -> MeasuredValue:
    ...


@overload
def value(val: SupportsFloat, unit: Unit, /) -> MeasuredValue:
    ...


@overload
def value(
    params: Iterable[PhysicalValue], function: Callable[..., Any], /
) -> NumericallyComputedValue:
    ...


@overload
def value(*args, **kwargs) -> PhysicalValue:
    ...


def value(*args, **kwargs) -> PhysicalValue:
    if len(args) == 0:
        raise ValueError("Must provide at least one argument.")

    if len(args) == 1:
        (arg,) = args
        if len(kwargs) == 0:
            if isinstance(arg, PhysicalValue):
                return arg
            else:
                return MeasuredValue.interpret(arg)
        elif isinstance(arg, str) and all(
            isinstance(v, PhysicalValue) for v in kwargs.values()
        ):
            return SymbolicallyComputedValue(arg, **kwargs)

    elif len(args) == 2:
        arg1, arg2 = args
        if isinstance(arg1, SupportsFloat) and (
            arg2 is None or isinstance(arg2, SupportsFloat)
        ):
            return MeasuredValue(arg1, unc=arg2)
        elif isinstance(arg1, SupportsFloat) and is_unit(arg2):
            return MeasuredValue(arg1, unit=arg2)
        elif isinstance(arg1, Iterable) and callable(arg2):
            return NumericallyComputedValue(arg1, arg2)

    elif len(args) == 3:
        arg1, arg2, arg3 = args
        if (
            isinstance(arg1, SupportsFloat)
            and (arg2 is None or isinstance(arg2, SupportsFloat))
            and is_unit(arg3)
        ):
            return MeasuredValue(arg1, unc=arg2, unit=arg3)

    elif len(args) == 4:
        arg1, arg2, arg3, arg4 = args
        if (
            isinstance(arg1, SupportsFloat)
            and (arg2 is None or isinstance(arg2, SupportsFloat))
            and is_unit(arg3)
            and (arg3 is None or isinstance(arg4, SupportsInt))
        ):
            return MeasuredValue(arg1, arg2, arg3, arg4)

    raise TypeError(f"Invalid arguments {format_args(*args, **kwargs)}.")


def _physical_value_compute_by(
    params: tuple[Var, ...], func: Callable[..., Any] | str
) -> Callable[..., PhysicalValue]:
    if isinstance(func, str):
        return lambda *args: SymbolicallyComputedValue(
            func, **{p.label.ascii: arg for p, arg in zip(params, args)}
        )
    else:
        return lambda *args: NumericallyComputedValue(args, func)


physical_value_var_type = VarType[PhysicalValue](
    PhysicalValue,
    "physical_value",
    default=lambda: math.nan,
    parse=lambda raw: value(raw),
    compute_by=_physical_value_compute_by,
    get_plot_value=lambda v: v.val,
)

VarType.register(physical_value_var_type)
