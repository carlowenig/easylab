from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
import math
import re
import sys
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    get_args,
    overload,
)
from typing_extensions import Self
import numpy as np
import sympy

from ..lang import TextInput, Text, lang, is_text_input


dim_symbols = {
    "length": Text("L"),
    "mass": Text("M"),
    "time": Text("T"),
    "current": Text("I"),
    "temperature": Text("Θ", latex="\\theta"),
    "amount_of_substance": Text("N"),
    "luminous_intensity": Text("J"),
}


# @dataclass(frozen=True)
# class Dim:
#     length: Fraction = Fraction(0)
#     mass: Fraction = Fraction(0)
#     time: Fraction = Fraction(0)
#     current: Fraction = Fraction(0)
#     temperature: Fraction = Fraction(0)
#     amount_of_substance: Fraction = Fraction(0)
#     luminous_intensity: Fraction = Fraction(0)

#     @property
#     def L(self):
#         return self.length

#     @property
#     def M(self):
#         return self.mass

#     @property
#     def T(self):
#         return self.time

#     @property
#     def Θ(self):
#         return self.temperature

#     @property
#     def I(self):
#         return self.current

#     @property
#     def N(self):
#         return self.amount_of_substance

#     @property
#     def J(self):
#         return self.luminous_intensity

#     @property
#     def exponents(self):
#         return {
#             "length": self.length,
#             "mass": self.mass,
#             "time": self.time,
#             "current": self.current,
#             "temperature": self.temperature,
#             "amount_of_substance": self.amount_of_substance,
#             "luminous_intensity": self.luminous_intensity,
#         }

#     @property
#     def is_one(self) -> bool:
#         return self == Dim()

#     def __str__(self):
#         s = "Dim("

#         for name, exp in self.exponents.items():
#             symbol = dim_symbols[name]

#             if exp == Fraction(1):
#                 s += f"{symbol} "
#             elif exp != Fraction(0):
#                 s += f"{symbol}^{exp} "

#         return s.strip() + ")"

#     def __add__(self, other: Any) -> Dim:
#         if other != self:
#             raise ValueError("Cannot add two different dimensions.")
#         return self

#     def __radd__(self, other: Any) -> Dim:
#         return self + other

#     def __sub__(self, other: Any) -> Dim:
#         if other != self:
#             raise ValueError("Cannot subtract two different dimensions.")
#         return self

#     def __rsub__(self, other: Any) -> Dim:
#         return self - other

#     def __neg__(self) -> Dim:
#         return self

#     def __mul__(self, other: Any) -> Dim:
#         if not isinstance(other, Dim):
#             return self

#         return Dim(
#             self.length + other.length,
#             self.mass + other.mass,
#             self.time + other.time,
#             self.current + other.current,
#             self.temperature + other.temperature,
#             self.amount_of_substance + other.amount_of_substance,
#             self.luminous_intensity + other.luminous_intensity,
#         )

#     def __rmul__(self, other: Any) -> Dim:
#         return self * other

#     def __pow__(self, exp: Any) -> Dim:
#         exp = Fraction(exp)
#         return Dim(
#             self.length * exp,
#             self.mass * exp,
#             self.time * exp,
#             self.current * exp,
#             self.temperature * exp,
#             self.amount_of_substance * exp,
#             self.luminous_intensity * exp,
#         )

#     def __truediv__(self, other: Any) -> Dim:
#         return self * other ** -1

#     def __rtruediv__(self, other: Any) -> Dim:
#         return other * self ** -1


def expr_text(expr: sympy.Expr) -> Text:
    if hasattr(expr, "text"):
        return cast(Text, getattr(expr, "text"))
    else:
        return Text(
            sympy.pretty(expr),
            unicode=sympy.pretty(expr, unicode=True),
            latex=sympy.latex(expr),
        )


class Parsable:
    @classmethod
    def _parse_input(cls, input: Any) -> Self | None:
        return None

    @classmethod
    def parse(cls, input: Any) -> Self:
        if isinstance(input, cls):
            return input

        result = cls._parse_input(input)

        if result is None:
            raise ValueError(f"Cannot parse {input} as {cls.__name__}.")

        return result


class MathObject(sympy.AtomicExpr, Parsable):
    # label: Text

    is_commutative = True
    is_real = True
    is_number = False
    is_nonzero = True

    label: Text | None
    expr: sympy.Expr | None

    def __new__(cls, label_or_expr: Any, expr: Any = None, /):
        if is_text_input(label_or_expr):
            label = label_or_expr
        else:
            label = None
            if expr is not None:
                raise ValueError(
                    f"Cannot specify second argument ({expr}), when first argument ({label_or_expr}) is not a label."
                )
            expr = label_or_expr

        if label is None and expr is None:
            raise ValueError("Must specify label or expr.")

        obj = cast(MathObject, super().__new__(cls))
        obj.label = Text.parse(label) if label is not None else None
        obj.expr = sympy.sympify(expr) if expr is not None else None
        return obj

    @property
    def text(self):
        return self.label or expr_text(cast(sympy.Expr, self.expr))

    @property
    def name(self):
        return self.text.default

    def _latex(self, printer):
        return self.text.latex

    def _eval_subs(self, old, new):
        if isinstance(new, MathObject) and self != old:
            return self

    @property
    def free_symbols(self):
        """Return free symbols from math object."""
        return set()

    # def __str__(self):
    #     return self.label.unicode

    # def __repr__(self):
    #     attrs_str = ", ".join(
    #         f"{name}={repr(value)}" for name, value in self.__dict__.items()
    #     )

    #     return f"{type(self).__name__}({attrs_str})"


def math_objects(expr: Any) -> set[MathObject]:
    return sympy.sympify(expr).atoms(MathObject)


def expr_type(expr: Any):
    object_types = {type(o) for o in math_objects(expr)}

    if len(object_types) == 1:
        return object_types.pop()
    elif len(object_types) == 0:
        return None
    else:
        raise ValueError(
            f"Expression {expr} has multiple types: {[t.__name__ for t in object_types]}."
        )


def is_expr_type(expr: Any, type_: type):
    _expr_type = expr_type(expr)
    if _expr_type is None:
        return False
    return issubclass(_expr_type, type_)


def check_expr_type(expr: Any, expected_type: type, *, allow_none: bool = True):
    t = expr_type(expr)
    if t is None:
        if allow_none:
            return
        else:
            raise ValueError(
                f"Expression {expr} has no type. Expected {expected_type.__name__}."
            )
    elif not issubclass(t, expected_type):
        raise ValueError(
            f"Expression {expr} has type {t.__name__}. Expected {expected_type.__name__}."
        )


_T = TypeVar("_T")


def find_objects_by_type(expr: sympy.Expr, type_: type[_T]) -> set[_T]:
    return {o for o in math_objects(expr) if isinstance(o, type_)}


def replace_objects(expr: sympy.Expr, replacer: Callable[[MathObject], Any]):
    for obj in math_objects(expr):
        # print("replace", obj, "->", replacer(obj), "in", expr)
        expr = expr.xreplace({obj: replacer(obj)})  # type: ignore
        # print("  =>", expr)

    return expr


def evaluate_objects(expr: sympy.Expr, replacer: Callable[[MathObject], Any]):
    # print(expr, "|->", replace_objects(expr, replacer))
    f = sympy.lambdify(
        [],
        replace_objects(expr, replacer),
        modules=["numpy", sys.modules[__name__]],
    )
    return f()


# class Dim(MathObject):
#     pass


# def simplify_dim(dim: Any):
#     return sympy.expand_power_base(sympy.simplify(dim))


# class dims(SimpleNamespace):
#     length = L = Dim("L")
#     mass = M = Dim("M")
#     time = T = Dim("T")
#     electric_current = I = Dim("I")
#     temperature = Θ = Dim("Θ")
#     amount_of_substance = N = Dim("N")
#     luminous_intensity = J = Dim("J")


def expr_equal(a: Any, b: Any):
    diff_expr = sympy.simplify(sympy.sympify(a) - sympy.sympify(b))
    return diff_expr == 0


DimLike = Any


class Dim(MathObject):
    def __new__(cls, label_or_expr: Any = None, expr: Any = None, /):
        if label_or_expr is None and expr is None:
            return Dim(None, 1)

        return super().__new__(cls, label_or_expr, expr)

    @classmethod
    def _parse_input(cls, input: Any) -> Self | None:
        if isinstance(input, str):
            expr = 1
            matches = re.findall(r"([MLTΘNJ])(?:\^(-?\d+))?", input)

            for match in matches:
                if len(match) == 1:
                    (name,) = match
                    exp = 1
                elif len(match) == 2:
                    name, exp_str = match
                    exp = int(exp_str)
                else:
                    raise ValueError(f"Invalid match: {match}")

                dim = dims._search_value(name)

                if dim is None:
                    raise ValueError(f"Unknown Dim: {name}")

                expr *= dim ** exp

            return cls(None, expr)


class Collection:
    _values: dict[str, Any] = {}

    def __init_subclass__(cls) -> None:
        for name, value in cls.__dict__.items():
            if not name.startswith("_"):
                cls._check_value(name, value)
                cls._values[name] = value

    @classmethod
    def _check_value(cls, name: str, value: Any) -> None:
        pass

    @classmethod
    def _get(cls, name: str):
        return cls._values.get(name)

    @classmethod
    def _names_of(cls, value: Any):
        return {name for name, v in cls._values.items() if v is value}

    @classmethod
    def _aliases(cls, name: str) -> set[str]:
        value = cls._get(name)
        return cls._names_of(value).difference({name})

    @classmethod
    def _search(cls, query: str):
        value = cls._get(query)
        if value is not None:
            return query, value

        for name, value in cls._values.items():
            if hasattr(value, "name") and getattr(value, "name") == query:
                return name, value

            if hasattr(value, "label") and cast(Text, getattr(value, "label")).matches(
                query
            ):
                return name, value

        return None

    @classmethod
    def _search_name(cls, query: str):
        name, _ = cls._search(query) or (None, None)
        return name

    @classmethod
    def _search_value(cls, query: str):
        _, value = cls._search(query) or (None, None)
        return value


class TypedCollection(Collection, Generic[_T]):
    _values: dict[str, _T] = {}

    @property
    @classmethod
    def _type(cls) -> type[_T]:
        return get_args(cls)[0]

    @classmethod
    def _get(cls, name: str):
        return cast(_T, super()._get(name))

    @classmethod
    def _search(cls, query: str) -> tuple[str, _T] | None:
        return cast(Optional[tuple[str, _T]], super()._search(query))

    @classmethod
    def _check_value(cls, name: str, value: Any) -> None:
        if not isinstance(value, cls._type):
            raise ValueError(f"Collection value {name} is not of type {cls._type}.")


class dims(TypedCollection[Dim]):
    mass = M = Dim("M")
    length = L = Dim("L")
    time = T = Dim("T")
    electric_current = I = Dim("I")
    temperature = Θ = Dim("Θ")
    amount_of_substance = N = Dim("N")
    luminous_intensity = J = Dim("J")


s_value = sympy.symbols("s_value")


UnitLike = Any


class Unit(MathObject):
    dim: DimLike
    value_expr: sympy.Expr

    def __new__(
        cls,
        label_or_expr: Any = None,
        expr: Any = None,
        *,
        dim: DimLike = 1,
        value_expr: Any = s_value,
    ):
        if label_or_expr is None and expr is None:
            return units.one

        obj = cast(Unit, super().__new__(cls, label_or_expr, expr))

        if obj.expr is not None:
            if dim != 1 or value_expr != s_value:
                raise Warning("Ignoring dim and value_expr since expr was given.")

            check_expr_type(obj.expr, Unit)

            dim = evaluate_objects(obj.expr, lambda o: cast(Unit, o).dim)

            if expr_type(dim) is None:
                dim = 1

            check_expr_type(dim, Dim)

            value_expr = replace_objects(obj.expr, lambda o: cast(Unit, o).value_expr)

        check_expr_type(dim, Dim)
        obj.dim = dim
        obj.value_expr = sympy.simplify(value_expr)
        return obj

    # @staticmethod
    # def from_expr(label: TextInput, expr_: UnitLike):
    #     expr: sympy.Expr = sympy.sympify(expr_)
    #     check_expr_type(expr, Unit)

    #     dim = evaluate_objects(expr, lambda o: cast(Unit, o).dim)

    #     if expr_type(dim) is None:
    #         dim = 1
    #     # print("dim of", label, "is", dim, "type", expr_type(dim))

    #     check_expr_type(dim, Dim)

    #     value_expr = sympy.simplify(
    #         replace_objects(expr, lambda o: cast(Unit, o).value_expr)
    #     )

    #     return Unit(label, dim=dim, value_expr=value_expr)


def _convert_from_base(value: float, to_unit: Unit) -> float:
    return to_unit.value_expr.subs(s_value, value).evalf()


def _convert_to_base(value: float, from_unit: Unit) -> float:
    results = sympy.solve(from_unit.value_expr - value, s_value)

    return sympy.N(results[0])


def convert(value: float, from_: UnitLike, to: UnitLike):
    from_unit = Unit(None, from_)
    to_unit = Unit(None, to)

    if from_unit.dim != to_unit.dim:
        raise ValueError(
            f"Cannot convert from unit {from_} to {to}, because they have different dimensions ({from_unit.dim} and {to_unit.dim}, respectively)."
        )

    return _convert_from_base(_convert_to_base(value, from_), to)


class units(TypedCollection[Unit]):
    # ------ Base units ------

    one = Unit("1")

    meter = meters = metre = metres = m = Unit("m", dim=dims.L)
    """Meter (m). SI base unit of length."""

    kilogram = kilograms = kg = Unit("kg", dim=dims.M)
    """Kilogram (kg). SI base unit of mass."""

    second = seconds = s = Unit("s", dim=dims.T)
    """Second (s). SI base unit of time."""

    ampere = A = Unit("A", dim=dims.I)
    """Ampere (A). SI base unit of electric current."""

    kelvin = K = Unit("K", dim=dims.Θ)
    """Kelvin (K). SI base unit of temperature."""

    mole = moles = mol = Unit("mol", dim=dims.N)
    """Mole (mol). SI base unit of amount of substance."""

    candela = cd = Unit("cd", dim=dims.J)
    """Candela (cd). SI base unit of luminous intensity."""

    # ------ Prefixes ------
    deca = Unit("da", 10)
    """Deca (da). Prefix for 10."""

    hecto = Unit("h", 100)
    """Hecto (h). Prefix for 100."""

    kilo = Unit("kilo", 1e3)
    """Kilo. SI prefix for 10^3."""

    mega = Unit("mega", 1e6)
    """Mega. SI prefix for 10^6."""

    giga = Unit("giga", 1e9)
    """Giga. SI prefix for 10^9."""

    tera = Unit("tera", 1e12)
    """Tera. SI prefix for 10^12."""

    peta = Unit("peta", 1e15)
    """Peta. SI prefix for 10^15."""

    exa = Unit("exa", 1e18)
    """Exa. SI prefix for 10^18."""

    zetta = Unit("zetta", 1e21)
    """Zetta. SI prefix for 10^21."""

    yotta = Unit("yotta", 1e24)
    """Yotta. SI prefix for 10^24."""

    deci = Unit("deci", 0.1)
    """Deci. SI prefix for 0.1."""

    centi = Unit("centi", 0.01)
    """Centi. SI prefix for 0.01."""

    milli = Unit("milli", 1e-3)
    """Milli. SI prefix for 10^-3."""

    micro = Unit("micro", 1e-6)
    """Micro. SI prefix for 10^-6."""

    nano = Unit("nano", 1e-9)
    """Nano. SI prefix for 10^-9."""

    pico = Unit("pico", 1e-12)
    """Pico. SI prefix for 10^-12."""

    femto = Unit("femto", 1e-15)
    """Femto. SI prefix for 10^-15."""

    atto = Unit("atto", 1e-18)
    """Atto. SI prefix for 10^-18."""

    zepto = Unit("zepto", 1e-21)
    """Zepto. SI prefix for 10^-21."""

    yocto = Unit("yocto", 1e-24)
    """Yocto. SI prefix for 10^-24."""

    # ------ Scaled units ------

    gram = grams = g = Unit("g", kilogram / kilo)
    """Gram (g). 1g = 10^-3 kg."""

    milligram = milligrams = mg = Unit("mg", milli * gram)
    """Milligram (mg). 1mg = 10^-3 g."""

    microgram = micrograms = μg = Unit("μg", micro * gram)
    """Microgram (μg). 1μg = 10^-6 g."""

    nanogram = nanograms = ng = Unit("ng", nano * gram)
    """Nanogram (ng). 1ng = 10^-9 g."""

    decimeter = decimeters = dm = Unit("dm", deci * meter)
    """Decimeter (dm). 1dm = 10^-1 m."""

    centimeter = centimeters = cm = Unit("cm", centi * meter)
    """Centimeter (cm). 1cm = 10^-2 m."""

    millimeter = millimeters = mm = Unit("mm", milli * meter)
    """Millimeter (mm). 1mm = 10^-3 m."""

    micrometer = micrometers = μm = Unit("μm", micro * meter)
    """Micrometer (μm). 1μm = 10^-6 m."""

    nanometer = nanometers = nm = Unit("nm", nano * meter)
    """Nanometer (nm). 1nm = 10^-9 m."""

    picometer = picometers = pm = Unit("pm", pico * meter)
    """Picometer (pm). 1pm = 10^-12 m."""

    femtometer = femtometers = fm = Unit("fm", femto * meter)
    """Femtometer (fm). 1fm = 10^-15 m."""

    attometer = attometers = am = Unit("am", atto * meter)
    """Attometer (am). 1am = 10^-18 m."""

    # ------ Derived units ------

    hertz = Hz = Unit("Hz", 1 / second)
    """Hertz (Hz). Derived SI unit for frequency. 1Hz = 1/s."""

    square_meter = square_meters = square_metre = square_metres = m2 = Unit(
        "m2", meter ** 2
    )
    """Square meter (m2)."""

    meters_per_second = metres_per_second = mps = Unit("m/s", meter / second)
    """Metres per second (m/s). Derived SI unit for velocity."""

    meters_per_second_squared = metres_per_second_squared = mps2 = Unit(
        "m/s²", meter / (second ** 2)
    )
    """Metres per second squared (m/s²). Derived SI unit for acceleration."""

    newton = N = Unit("N", kilogram * meter / (second ** 2))
    """Newton (N). Derived SI unit for force. 1N = 1kg·m/s²."""

    joule = J = Unit("J", newton * meter)
    """Joule (J). Derived SI unit for energy. 1J = 1N·m."""

    watt = watts = W = Unit("W", joule / second)
    """Watt (W). Derived SI unit for power. 1W = 1J/s."""

    pascal = Pa = Unit("Pa", newton / (meter ** 2))
    """Pascal (Pa). Derived SI unit for pressure. 1Pa = 1N/m²."""

    joule_per_kelvin = J_K = Unit("J/K", joule / kelvin)
    """Joule per Kelvin (J/K). Derived SI unit for heat capacity."""

    volt = volts = V = Unit("V", watt / ampere)
    """Volt (V). Derived SI unit for electric potential. 1V = 1W/A."""

    coulomb = C = Unit("C", ampere * second)
    """Coulomb (C). Derived SI unit for electric charge. 1C = 1A·s."""

    farad = F = Unit("F", coulomb / volt)
    """Farad (F). Derived SI unit for electric capacitance. 1F = 1C/V."""

    ohm = Unit(lang.Omega, volt / ampere)
    """Ohm (Ω). Derived SI unit for electric resistance."""

    henry = H = Unit("H", ohm * ampere)
    """Henry (H). Derived SI unit for inductance."""

    weber = Wb = Unit("Wb", volt * second)
    """Weber (Wb). Derived SI unit for magnetic flux."""

    tesla = T = Unit("T", weber / ampere)
    """Tesla (T). Derived SI unit for magnetic field."""

    henry_per_meter = H_m = Unit("H/m", henry / meter)
    """Henry per meter (H/m). Derived SI unit for magnetic flux density."""

    degrees_celsius = celsius = degC = Unit(lang.degree + "C", kelvin - 273.15)
    """Degrees Celsius (°C). Derived unit for temperature. T[°C] = T[K] - 273.15."""

    # ------ Other units ------

    radian = radians = rad = Unit("rad")
    """Radian (rad). Derived SI unit for angle."""

    degree = degrees = deg = Unit(lang.degree, radian / math.pi * 180)
    """Degree (°). Derived SI unit for angle. 1° = pi/180 rad."""

    steradian = steradians = sr = Unit("sr", radian ** 2)
    """Steradian (sr). Derived SI unit for solid angle."""

    # ------ Derived units with prefixes ------

    kilometer = kilometre = km = Unit("km", kilo * meter)

    kilohertz = kHz = Unit("kHz", kilo * hertz)
    """Kilohertz (kHz). Derived SI unit for frequency."""

    megahertz = MHz = Unit("MHz", mega * hertz)


# default_sample_width = 1e-8


# class Constraint(MathObject):
#     def __init__(self, *, sample_width: float = default_sample_width) -> None:
#         self.sample_width = sample_width
#         self.__check_normalized()

#     @abstractmethod
#     def prob_density(self, x: float) -> float:
#         pass

#     @overload
#     def prob(self, value: float, /) -> float:
#         ...

#     @overload
#     def prob(self, lower: float, upper: float, /) -> float:
#         ...

#     def prob(self, value_or_lower: float, upper: float | None = None, /) -> float:
#         if upper is None:
#             return self.prob_density(value_or_lower) * self.sample_width
#         else:
#             return sum(
#                 self.prob_density(x) * self.sample_width
#                 for x in np.arange(value_or_lower, upper, self.sample_width)
#             )

#     def __check_normalized(self):
#         norm = self.prob(-np.inf, np.inf)

#         if abs(norm - 1) > self.sample_width:
#             raise ValueError(f"Constraint is not normalized. Norm was {norm}.")


# class EqualConstraint(Constraint):
#     def __init__(self, value: float, **kwargs):
#         self.value = value
#         super().__init__(**kwargs)

#     def prob_density(self, x: float):
#         return 1 if abs(self.value - x) < self.sample_width else 0


# class IntervalConstraint(Constraint):
#     def __init__(self, lower: float = -math.inf, upper: float = math.inf, **kwargs):
#         assert upper > lower
#         self.lower = lower
#         self.upper = upper
#         super().__init__(**kwargs)

#     @property
#     def width(self):
#         return self.upper - self.lower

#     def prob_density(self, x: float):
#         if x < self.lower or x > self.upper:
#             return 0
#         else:
#             return 1 / self.width


# class GaussianConstraint(Constraint):
#     def __init__(self, mean: float, std: float, **kwargs):
#         self.mean = mean
#         self.std = std
#         super().__init__(**kwargs)

#     def prob_density(self, x: float):
#         return np.exp(-((x - self.mean) ** 2) / (2 * self.std ** 2)) / (
#             np.sqrt(2 * np.pi) * self.std
#         )

# class SumConstraint(Constraint):
#     def __init__(self, *constraints: Constraint, **kwargs):
#         self.constraints = constraints
#         super().__init__(**kwargs)

#     def prob_density(self, x: float):
#         return sum(c.prob_density(x) for c in self.constraints)


class Value(MathObject):
    mean: float
    err: float
    unit: UnitLike
    prec: int

    def __new__(
        cls,
        label_or_expr: Any,
        expr: Any = None,
        *,
        mean: float,
        err: float,
        unit: UnitLike = 1,
        prec: int | Literal["infer"] = "infer",
    ) -> "Value":

        obj = cast(Value, super().__new__(cls, label_or_expr, expr))

        if obj.expr is not None:
            check_expr_type(obj.expr, Var)

            inferred_unit = Unit(
                None, evaluate_objects(obj.expr, lambda o: cast(Var, o).unit)
            )
            if unit == "infer":
                _unit = inferred_unit
            else:
                _unit = Unit(None, unit)
                if _unit.dim != inferred_unit.dim:
                    raise ValueError(
                        f"Cannot express {expr} in unit {_unit}, since it has unit {inferred_unit}."
                    )
        else:
            _unit = Unit(None, unit)

        obj.unit = unit
        obj.prec = prec
        return obj


VarLike = Any


class Var(MathObject):
    unit: UnitLike
    prec: int | Literal["infer"]

    def __new__(
        cls,
        label_or_expr: Any,
        expr: Any = None,
        *,
        unit: UnitLike | Literal["infer"] = "infer",
        prec: int | Literal["infer"] = "infer",
    ):
        obj = cast(Var, super().__new__(cls, label_or_expr, expr))

        if obj.expr is not None:
            check_expr_type(obj.expr, Var)

            inferred_unit = Unit(
                None, evaluate_objects(obj.expr, lambda o: cast(Var, o).unit)
            )
            if unit == "infer":
                _unit = inferred_unit
            else:
                _unit = Unit(None, unit)
                if _unit.dim != inferred_unit.dim:
                    raise ValueError(
                        f"Cannot express {expr} in unit {_unit}, since it has unit {inferred_unit}."
                    )
        else:
            _unit = Unit(None, unit) if unit != "infer" else units.one

        obj.unit = _unit
        obj.prec = prec
        return obj


# class Computed(Var):
#     expr: sympy.Expr

#     def __new__(
#         cls,
#         label: TextInput,
#         expr_: VarLike,
#         *,
#         unit: UnitLike | Literal["infer"] = "infer",
#         prec: int | Literal["infer"] = "infer",
#     ):
#         expr: sympy.Expr = sympy.sympify(expr_)
#         check_expr_type(expr, Var)

#         inferred_unit = evaluate_objects(expr, lambda o: cast(Var, o).unit)
#         if unit == "infer":
#             unit = inferred_unit
#         elif Unit.from_expr("?", unit).dim != Unit.from_expr("?", inferred_unit).dim:
#             raise ValueError(
#                 f"Unit {unit} cannot be used for computed variable {label}, since its expression {expr} has unit {inferred_unit}."
#             )

#         print(Unit.from_expr("?", unit).dim)
#         print(Unit.from_expr("?", inferred_unit).dim)

#         obj = cast(Computed, super().__new__(cls, label, unit=unit, prec=prec))
#         obj.expr = expr
#         return obj


print(Unit(5 * units.meter).dim)


x = Var("x", unit=units.m)
t = Var("t", unit=units.s)

v = Var("v", x / t, unit=units.km / units.s)

print(v.unit, v.prec)
