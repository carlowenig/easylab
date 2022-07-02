from dataclasses import dataclass
from decimal import Decimal
from numbers import Number
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    Sequence,
    SupportsFloat,
    SupportsInt,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from typing_extensions import TypeGuard

import numpy as np
import numpy.typing as npt


UnitInput = Union[str, "Unit", None]


class Unit:
    name: str

    def __init__(self, name: str = ""):
        self.name = name

    @staticmethod
    def parse(input: UnitInput) -> "Unit":
        if input is None:
            return Unit()
        elif isinstance(input, Unit):
            return input
        elif isinstance(input, str):
            return Unit(input)
        else:
            raise ValueError(f"Cannot parse {input} as Unit.")

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Unit({self.name})"


class Measured:
    value: float
    error: float
    unit: Unit

    def __init__(
        self,
        value: float,
        *,
        error: float = 0,
        unit: Unit = Unit(""),
    ):
        self.value = value
        self.error = error
        self.unit = unit

    @staticmethod
    def parse(input: Any, **kwargs) -> "Measured":
        if isinstance(input, Measured):
            return input
        elif isinstance(input, SupportsFloat):
            return Measured(float(input), **kwargs)
        elif isinstance(input, tuple) and len(input) == 2:
            return Measured(input[0], error=input[1], **kwargs)
        else:
            raise ValueError(f"Cannot parse {input} as Measured.")

    def __str__(self):
        s = f"{self.value:.3g}"
        if self.error != 0:
            s += f" ± {self.error:.3g}"
        if self.unit.name != "":
            if self.error != 0:
                s = f"({s})"
            s += f"{self.unit.name}"
        return s


PreciseValueInput = Union[SupportsFloat, str]


def is_precise_value_input(input: Any) -> TypeGuard[PreciseValueInput]:
    return isinstance(input, (SupportsFloat, str))


PreciseInput = Union[PreciseValueInput, Tuple[PreciseValueInput, int], "Precise"]


def is_precise_input(input: Any) -> TypeGuard[PreciseInput]:
    return (
        isinstance(input, Precise)
        or is_precise_value_input(input)
        or (
            isinstance(input, tuple)
            and len(input) == 2
            and is_precise_value_input(input[0])
            and isinstance(input[1], int)
        )
    )


RoundingMethod = Literal["standard", "down", "up"]
PrecisionConstraint = Literal["tight", "loose"]


def parse_precise_value_input(
    input: PreciseValueInput,
) -> Tuple[float, int, PrecisionConstraint, Optional[RoundingMethod]]:

    value: float
    precision: int
    rounding: Optional[RoundingMethod] = None
    precision_constraint: PrecisionConstraint = "loose"

    if isinstance(input, str):
        input = input.strip()
        precision_constraint = "tight"
        if input.endswith("+"):
            rounding = "up"
            s = input[:-1]
        elif input.endswith("-"):
            rounding = "down"
            s = input[:-1]
        elif input.endswith("r"):
            rounding = "standard"
            s = input[:-1]
        else:
            s = input
        value = float(s)
    elif isinstance(input, SupportsFloat):
        # First remove trailing zeros, then remove trailing decimal point, if existent
        value = float(input)
        s = str(value).rstrip("0").rstrip(".")

    if "." not in s:
        # Value is an integer. We could use a negative precision here, but 0 will be more practical in most cases.
        precision = 0
    else:
        # Value is float
        parts = s.split(".")

        if len(parts) != 2:
            raise ValueError(
                f"Cannot infer precision of {s}. Invalid number of decimal points."
            )

        precision = len(parts[1])

    return value, precision, precision_constraint, rounding


def round_by_method(
    value: float, precision: int, method: Optional[RoundingMethod]
) -> float:
    if method is None:
        return float(value)
    elif method == "standard":
        return np.round(float(value), precision)
    elif method == "down":
        return np.floor(float(value) * 10 ** precision) / 10 ** precision
    elif method == "up":
        return np.ceil(float(value) * 10 ** precision) / 10 ** precision
    else:
        raise ValueError(f"Unknown rounding method '{method}'.")


class Precise(np.lib.mixins.NDArrayOperatorsMixin):
    __slots__ = ["value", "precision", "rounding"]

    value: float
    precision: int
    rounding: Optional[RoundingMethod]
    precision_constraint: PrecisionConstraint

    def __init__(
        self,
        value: PreciseValueInput,
        precision: Optional[int] = None,
        *,
        rounding: Optional[RoundingMethod] = None,
        precision_constraint: Optional[PrecisionConstraint] = None,
    ):
        self.rounding = rounding

        (
            parsed_value,
            parsed_precision,
            parsed_precision_constraint,
            parsed_rounding,
        ) = parse_precise_value_input(value)

        if precision_constraint is None:
            precision_constraint = "tight" if precision is not None else "loose"

        if precision is None:
            precision = parsed_precision
            precision_constraint = parsed_precision_constraint

        if rounding is None:
            rounding = parsed_rounding

        self.value = round_by_method(parsed_value, precision, rounding)
        self.precision = precision
        self.rounding = rounding
        self.precision_constraint = precision_constraint

    @staticmethod
    def parse(
        input: PreciseInput,
        *,
        rounding: Optional[RoundingMethod] = None,
    ):
        if isinstance(input, Precise):
            return input
        elif (
            isinstance(input, tuple)
            and len(input) == 2
            and is_precise_value_input(input[0])
            and isinstance(input[1], int)
        ):
            return Precise(input[0], precision=input[1], rounding=rounding)
        elif is_precise_value_input(input):
            return Precise(input, rounding=rounding)
        else:
            raise ValueError(f"Cannot parse {input} as Precise.")

    def set_precision(
        self, precision: int, *, rounding: Optional[RoundingMethod] = None
    ):
        return Precise(
            self.value, precision=precision, rounding=rounding or self.rounding
        )

    def __mod__(self, other: int):
        return self.set_precision(other)

    def __str__(self):
        if self.precision > 0:
            return f"{self.value:.{self.precision}f}"
        else:
            return str(int(self.value))

    def __repr__(self):
        s = str(self)

        if self.rounding == "standard":
            s += "r"
        elif self.rounding == "up":
            s += "+"
        elif self.rounding == "down":
            s += "-"

        s += f"[{self.precision}"
        if self.precision_constraint == "tight":
            s += "!"
        else:
            s += "~"
        s += "]"

        return s

    def __eq__(self, other: object) -> bool:
        try:
            other = Precise.parse(cast(PreciseInput, other))
            return (
                self.value == other.value
                and self.precision == other.precision
                and self.rounding == other.rounding
            )
        except ValueError:
            return False

    def __hash__(self) -> int:
        return hash((self.value, self.precision, self.rounding))

    def __format__(self, format_spec: str) -> str:
        return format(self.value, format_spec)

    # Implement __array__ method to allow numpy ufuncs
    def __array__(self, dtype: npt.DTypeLike = None) -> np.ndarray:
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        ufunc_str = ufunc.__name__ + "(" + ", ".join(map(str, inputs)) + ")"

        input_values = []
        precision = self.precision
        rounding = self.rounding
        for input in inputs:
            if isinstance(input, Precise):
                input_values.append(input.value)
                if precision is not None and input.precision != precision:
                    if (
                        input.precision_constraint == "loose"
                        and self.precision_constraint == "loose"
                    ):
                        # Both values have loose precision, so we can infer the precision of the result
                        precision = None
                    elif (
                        input.precision_constraint == "tight"
                        and self.precision_constraint == "tight"
                    ):
                        # Both values have tight precision, which means we can't infer the precision of the result
                        raise ValueError(
                            f"Cannot infer precision of operation {ufunc_str}, since at least to arguments have differing and tight precision."
                        )
                    elif (
                        input.precision_constraint == "tight"
                        and self.precision_constraint == "loose"
                    ):
                        # Input has tight precision, so use that precision
                        precision = input.precision

                if input.rounding is not None and input.rounding != rounding:
                    print(
                        f"[i] Found mismatching rounding methods {rounding} and {input.rounding} while performing {ufunc_str}. Falling back to standard."
                    )
                    rounding = "standard"
            else:
                input_values.append(input)

        return Precise(
            getattr(ufunc, method)(*input_values, **kwargs),
            precision=precision,
            rounding=rounding,
        )

    def __add__(self, other: PreciseInput) -> "Precise":
        return super().__add__(Precise.parse(other))

    def __radd__(self, other: PreciseInput) -> "Precise":
        return super().__radd__(Precise.parse(other))

    def __sub__(self, other: PreciseInput) -> "Precise":
        return super().__sub__(Precise.parse(other))

    def __rsub__(self, other: PreciseInput) -> "Precise":
        return super().__rsub__(Precise.parse(other))

    def __mul__(self, other: PreciseInput) -> "Precise":
        return super().__mul__(Precise.parse(other))

    def __rmul__(self, other: PreciseInput) -> "Precise":
        return super().__rmul__(Precise.parse(other))

    def __truediv__(self, other: PreciseInput) -> "Precise":
        return super().__truediv__(Precise.parse(other))

    def __rtruediv__(self, other: PreciseInput) -> "Precise":
        return super().__rtruediv__(Precise.parse(other))

    def __pow__(self, other: PreciseInput) -> "Precise":
        return super().__pow__(Precise.parse(other))

    def __rpow__(self, other: PreciseInput) -> "Precise":
        return super().__rpow__(Precise.parse(other))


UncertaintyInput = Union[PreciseInput, Tuple[PreciseInput, PreciseInput], "Uncertainty"]


class Uncertainty:
    __slots__ = ["positive", "negative"]

    positive: Precise
    negative: Precise

    def __init__(self, positive: PreciseInput = 0, negative: PreciseInput = 0):
        self.positive = Precise.parse(positive)
        self.negative = Precise.parse(negative)

    @staticmethod
    def symmetric(uncertainty: PreciseInput):
        return Uncertainty(uncertainty, uncertainty)

    @staticmethod
    def parse(input: UncertaintyInput):
        if isinstance(input, Uncertainty):
            return input
        elif isinstance(input, tuple) and len(input) == 2:
            return Uncertainty(input[0], cast(PreciseInput, input[1]))
        else:
            return Uncertainty.symmetric(input)

    @property
    def is_symmetric(self):
        return self.positive == self.negative

    def __str__(self):
        if self.is_symmetric:
            return f"±{self.positive:.3g}"
        else:
            return f"(+{self.positive:.3g} / -{self.negative:.3g})"

    def __eq__(self, other: object) -> bool:
        try:
            other = Uncertainty.parse(cast(UncertaintyInput, other))
            return self.positive == other.positive and self.negative == other.negative
        except ValueError:
            return False

    def __hash__(self) -> int:
        return hash((self.positive, self.negative))


UncertainInput = Union[SupportsFloat, "Uncertain"]


class Uncertain:
    __slots__ = ["mean", "uncertainty"]

    mean: float
    uncertainty: Uncertainty

    def __init__(
        self,
        mean: SupportsFloat,
        uncertainty: UncertaintyInput = 0,
    ):
        self.mean = float(mean)
        self.uncertainty = Uncertainty.parse(uncertainty)

    @staticmethod
    def parse(input: UncertainInput):
        if isinstance(input, Uncertain):
            return input
        elif isinstance(input, SupportsFloat):
            return Uncertain(float(input), 0)
        else:
            raise ValueError(f"Cannot parse {input} as Uncertain.")

    def __str__(self):
        s = f"{self.mean:.3g}"
        if self.uncertainty != 0:
            if self.uncertainty.is_symmetric:
                s = f"({s} ± {self.uncertainty.positive:.3g})"
            else:
                s = f"({s} +{self.uncertainty.positive:.3g} / -{self.uncertainty.negative:.3g})"
        return s

    def __eq__(self, other: object) -> bool:
        try:
            other = Uncertain.parse(cast(UncertainInput, other))
            return self.mean == other.mean and self.uncertainty == other.uncertainty
        except ValueError:
            return False

    def __hash__(self) -> int:
        return hash((self.mean, self.uncertainty))


T = TypeVar("T")

WithUnitInput = Union[T, Tuple[T, UnitInput], "WithUnit[T]"]


class WithUnit(Generic[T]):
    __slots__ = ["value", "unit"]

    value: T
    unit: Unit

    def __init__(self, value: T, unit: UnitInput = None):
        self.value = value
        self.unit = Unit.parse(unit)

    @staticmethod
    def parse(input: WithUnitInput[T]) -> "WithUnit[T]":
        if isinstance(input, WithUnit):
            return input
        elif isinstance(input, tuple) and len(input) == 2:
            return WithUnit(cast(T, input[0]), Unit.parse(input[1]))
        else:
            return WithUnit(input)

    def __eq__(self, other: object) -> bool:
        try:
            other = WithUnit.parse(cast(WithUnitInput, other))
            return self.value == other.value and self.unit == other.unit
        except ValueError:
            return False

    def __hash__(self) -> int:
        return hash((self.value, self.unit))

    def __str__(self):
        return f"{self.value}{self.unit.name}"


def measured(
    value: SupportsFloat, *, error: UncertaintyInput = 0, unit: UnitInput = None
):
    return WithUnit(Uncertain(value, error), unit)


@dataclass(frozen=True)
class Quantity:
    time: int = 0
    length: int = 0
    mass: int = 0
    temperature: int = 0
    amount_of_substance: int = 0
    current: int = 0
    luminous_intensity: int = 0

    def __mul__(self, other: "Quantity"):
        return Quantity(
            self.time + other.time,
            self.length + other.length,
            self.mass + other.mass,
            self.temperature + other.temperature,
            self.amount_of_substance + other.amount_of_substance,
            self.current + other.current,
            self.luminous_intensity + other.luminous_intensity,
        )

    def __pow__(self, exp: int):
        return Quantity(
            self.time * exp,
            self.length * exp,
            self.mass * exp,
            self.temperature * exp,
            self.amount_of_substance * exp,
            self.current * exp,
            self.luminous_intensity * exp,
        )

    def __truediv__(self, other: "Quantity"):
        return self * other ** -1


class Quantities:
    time = Quantity(time=1)
    length = Quantity(length=1)
    mass = Quantity(mass=1)
    temperature = Quantity(temperature=1)
    amount_of_substance = Quantity(amount_of_substance=1)
    current = Quantity(current=1)
    luminous_intensity = Quantity(luminous_intensity=1)

    velocity = length / time
    acceleration = velocity / time


ValueInput = Union[PreciseInput, "Value"]


class Value:
    mean: Precise
    uncertainty: Uncertainty
    quantity: Quantity

    floor: Precise
    ceil: Precise

    def __init__(
        self,
        mean: PreciseInput,
        uncertainty: UncertaintyInput = 0,
        quantity: Quantity = Quantity(),
    ):
        self.mean = Precise.parse(mean)
        self.uncertainty = Uncertainty.parse(uncertainty)
        self.quantity = quantity

        self.floor = self.mean - self.uncertainty.negative
        self.ceil = self.mean + self.uncertainty.positive

    @staticmethod
    def parse(input: ValueInput):
        if isinstance(input, Value):
            return input
        elif is_precise_input(input):
            return Value(input)
        else:
            raise ValueError(f"Cannot parse {input} as Value.")
