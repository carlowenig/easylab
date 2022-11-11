from abc import abstractmethod
import math
from types import SimpleNamespace
from typing import Any, Callable, Generic, Iterable, Literal, Optional, TypeVar, Union, cast
from typing_extensions import Self, TypeGuard
import numpy as np

import sympy
from ..lang import Text, TextInput, lang, HasText

_T = TypeVar("_T")

class Container(Generic[_T]):
    _value: _T
    _type: type[_T]
    _initial_value: _T

    def __init__(self, value: _T):
        self._value = value
        self._type = type(value)
        self._initial_value = value
    
    @property
    def value(self) -> _T:
        return self.__on_get__(self._value)

    @value.setter
    def value(self, input: _T):
        self._value = self.__on_set__(input)

    @property
    def type(self):
        return self._type

    def set(self, input: _T):
        self.value = input
        return self

    def reset(self):
        return self.set(self._initial_value)

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Container) and type(__o) == type(self) and self.value == __o.value

    def __hash__(self) -> int:
        return hash((type(self), self.value))

    def __str__(self) -> str:
        return f"(({str(self.value)} : {self._type.__name__}))"

    def __repr__(self) -> str:
        return f"Container[{self._type.__name__}]({repr(self.value)})"

    # Hooks
    def __on_set__(self, value: _T) -> _T:
        return value

    def __on_get__(self, value: _T) -> _T:
        return value


Evaluator = Literal["math", "numpy", "sympy"]


_E = TypeVar("_E", bound="MathEntity")

class MathEntity(HasText):
    @property
    def args(self) -> tuple["MathEntity", ...]:
        return ()

    @property
    def arg_count(self):
        return len(self.args)

    def _copy_with_args(self, *args: "MathEntity"):
        """Override this method. It is ensured that args has the correct length."""
        return self
    

    def copy_with_args(self, *args: "MathEntity"):
        if len(args) != self.arg_count:
            raise ValueError(f"{self.__class__.__name__} expects {self.arg_count} arguments, got {len(args)}.")
        return self._copy_with_args(*args)

    def copy(self):
        return self.copy_with_args(*self.args)

    @property
    def symbols(self) -> set["Symbol"]:
        if isinstance(self, Symbol):
            return {self}
        else:
            symbols: set[Symbol] = set()
            for arg in self.args:
                symbols.update(arg.symbols)
            return symbols

    def subs(self, values: dict["Symbol", "MathEntity"]) -> Self:
        return self

    @property
    def equivalent_variants(self) -> Iterable["MathEntity"]:
        return ()

    def deep_equivalent_variants(self, depth: int = 32) -> Iterable["MathEntity"]:
        yield from self.equivalent_variants

        if depth == 0:
            return

        for variant in self.equivalent_variants:
            yield from variant.deep_equivalent_variants(depth-1)


    def is_equivalent_to(self, other: "MathEntity", *, depth: int = 32) -> bool:
        if other == self:
            return True
        if other in self.deep_equivalent_variants(depth):
            return True

        args = self.args
        other_args = other.args

        if len(args) == len(other_args):
            for arg, other_arg in zip(args, other_args):
                if not arg.is_equivalent_to(other_arg, depth=depth-1):
                    return False

        return False

    def _eval_args(self, *arg_values: Any, evaluator: Evaluator) -> Any:
        raise NotImplementedError()

    def eval(self, values: dict["Symbol", Any], evaluator: Evaluator = "numpy") -> Any:
        arg_values = [arg.eval(values, evaluator) for arg in self.args]
        return self._eval_args(*arg_values, evaluator=evaluator)

    def __mul__(self, other: "MathEntity"):
        return Product(self, other)

    def __add__(self, other: "MathEntity"):
        return Sum(self, other)

    def __sub__(self, other: "MathEntity"):
        return Diff(self, other)

    def __pow__(self, exp: "MathEntity"):
        return Pow(self, exp)

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, MathEntity) and type(__o) == type(self) and list(self.args) == list(__o.args)

    def __hash__(self):
        return hash((type(self), *self.args))


_A = TypeVar("_A", bound=MathEntity)
_B = TypeVar("_B", bound=MathEntity)


class Symbol(MathEntity):
    label: Text
    values: dict[Evaluator, Any]

    def __init__(self, label: TextInput, values: dict[Evaluator, Any] = {}):
        self.label = Text.parse(label)
        self.values = values

    @property
    def text(self):
        return self.label

    @property
    def args(self) -> Iterable["MathEntity"]:
        return ()

    def eval(self, values: dict["Symbol", Any], evaluator: Evaluator = "numpy") -> Any:
        if self in values:
            return values[self]
        elif evaluator in self.values:
            return self.values[evaluator]
        else:
            raise ValueError(f"No value given for symbol {self}.")

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Symbol) and self.label == __o.label

class symbols(SimpleNamespace):
    e = Symbol("e", {"math": math.e, "numpy": np.e, "sympy": sympy.E})
    pi = Symbol("pi", {"math": math.pi, "numpy": np.pi, "sympy": sympy.pi})
    i = Symbol("i", {"math": 1j, "numpy": 1j, "sympy": sympy.I})


class Int(Symbol):
    value: int

    def __init__(self, value: int):
        self.value = value
        super().__init__(str(value), {"math": value, "numpy": value, "sympy": sympy.Integer(value)})


class Operation(MathEntity):
    args: tuple[MathEntity, ...]

    def __init__(self, *args: MathEntity) -> None:
        self.args = args

    def _copy_with_args(self, *args: "MathEntity"):
        return type(self)(*args)

class UnaryOperation(Operation, Generic[_A]):
    arg: _A

    def __init__(self, arg: _A) -> None:
        self.arg = arg
        super().__init__(arg)

class BinaryOperation(Operation, Generic[_A, _B]):
    a: _A
    b: _B

    # Class attributes
    commutative = False
    associative = False

    def __init__(self, a: _A, b: _B, /) -> None:
        self.a = a
        self.b = b
        super().__init__(a, b)
        
    def commute(self):
        return type(self)(self.b, self.a)

    @property
    def equivalent_variants(self) -> Iterable["MathEntity"]:
        yield from super().equivalent_variants

        if self.commutative:
            yield self.commute()
        if self.associative:
            if isinstance(self.a, BinaryOperation) and type(self.a) == type(self):
                yield self.copy_with_args(self.a.a, self.copy_with_args(self.a.b, self.b))
            if isinstance(self.b, BinaryOperation) and type(self.b) == type(self):
                yield self.copy_with_args(self.copy_with_args(self.a, self.b.a), self.b.b)

    

class InfixOperation(BinaryOperation[_A, _B]):
    # Class attributes
    separator = Text(" (?) ")

    @property
    def text(self) -> Text:
        return self.a.text + self.separator + self.b.text


class Product(InfixOperation[_A, _B]):
    separator = " " + lang.cdot + " "
    commutative = True
    associative = True

    def _eval_args(self, a: Any, b: Any, *, evaluator: Evaluator) -> Any:
        return a * b

class Neg(UnaryOperation[_A]):
    @property
    def text(self) -> Text:
        return "-" + self.arg.text

    def _eval_args(self, a: Any, *, evaluator: Evaluator) -> Any:
        return -a
 
    @property
    def equivalent_variants(self) -> Iterable["MathEntity"]:
        yield from super().equivalent_variants

        yield Product(Int(-1), self)

class Sum(InfixOperation[_A, _B]):  
    separator = Text(" + ")
    commutative = True
    associative = True

    def _eval_args(self, a: Any, b: Any, *, evaluator: Evaluator) -> Any:
        return a + b

class Diff(InfixOperation[_A, _B]):  
    separator = Text(" - ")
    commutative = False
    associative = True

    def _eval_args(self, a: Any, b: Any, *, evaluator: Evaluator) -> Any:
        return a + b

class Pow(BinaryOperation[_A, _B]):
    base: _A
    exp: _B

    def __init__(self, base: _A, exp: _B):
        self.base = base
        self.exp = exp
        super().__init__(base, exp)

    @property
    def text(self) -> Text:
        return self.base.text.superscript(self.exp.text)

    def _eval_args(self, base: Any, exp: Any, *, evaluator: Evaluator) -> Any:
        return base ** exp

        
    @property
    def equivalent_variants(self) -> Iterable["MathEntity"]:
        yield from super().equivalent_variants

        # x^{y + z} = x^y * x^z
        if isinstance(self.exp, Sum):
            yield Product(Pow(self.base, self.exp.a), Pow(self.base, self.exp.b))

        # e^x = exp(x)
        if self.base.is_equivalent_to(symbols.e):
            yield Exp(self.exp)



class FunctionalOperation(Operation):
    name: str = "?"
    eval_functions: dict[Evaluator, Callable[..., Any]] = {}

    @property
    def text(self) -> Text:
        return lang.mathrm(self.name) + lang.par(Text(", ").join(arg.text for arg in self.args))
        
    def _eval_args(self, arg: Any, *, evaluator: Evaluator) -> Any:
        if evaluator in self.eval_functions:
            return self.eval_functions[evaluator](arg)
        else:
            raise ValueError(f"{self} cannot be evaluated by {evaluator}.")

    
class Exp(UnaryOperation[_A], FunctionalOperation):
    name = "exp"
    eval_functions = {
        "math": math.exp,
        "numpy": np.exp,
        "sympy": sympy.exp,
    }

    @property
    def equivalent_variants(self) -> Iterable["MathEntity"]:
        yield from super().equivalent_variants

        yield Pow(symbols.e, self.arg)

        if isinstance(self.arg, Log):
            yield self.arg.arg

class Log(UnaryOperation[_A], FunctionalOperation):
    name = "log"
    eval_functions = {
        "math": math.log,
        "numpy": np.log,
        "sympy": sympy.log,
    }

    @property
    def equivalent_variants(self) -> Iterable["MathEntity"]:
        yield from super().equivalent_variants

        yield Pow(symbols.e, self.arg)


class Dim(HasText):
    _powers: dict[str, int]

    def __init__(self, **powers: int):
        self._powers = powers

    def __getitem__(self, key: str) -> int:
        return self._powers[key]
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Dim) and self._powers == other._powers

    def __hash__(self) -> int:
        return hash(tuple(self._powers.items()))

    @property
    def text(self):
        return lang.cdot.join(Text(d).superscript(str(exp)) for d, exp in self._powers.items())

    def __mul__(self, other: "Dim"):
        powers: dict[str, int] = {}


class Unit(HasText):
    @abstractmethod
    def convert_to_base(self, value: float) -> float:
        pass

    @abstractmethod
    def convert_from_base(self, base_value: float) -> float:
        pass

    def supports_conversion_to(self, other: "Unit") -> bool:
        """Always override this method instead of is_convertible_to."""
        return False

    @property
    def value_separator(self) -> Text:
        return Text("")

    def format_value(
        self, value: float, precision: Optional[int] = None, decimal: str = ","
    ):
        value_text = lang.number(value, precision=precision, decimal=decimal)
        return value_text + self.value_separator + self.text

    def simplify(self) -> "Unit":
        return self

    def is_convertible_to(self, other: "Unit") -> bool:
        """Never override this method. Use supports_conversion_to instead."""
        return self.supports_conversion_to(other) or other.supports_conversion_to(self)

    def convert(self, value: float, to: "Unit"):
        if not self.is_convertible_to(to):
            raise ValueError(f"Cannot convert from unit {self} to {to}.")
        return to.convert_from_base(self.convert_to_base(value))

    @property
    def dependencies(self) -> Iterable["Unit"]:
        return []

    def __mul__(self, other: Union["Unit", float]):
        if isinstance(other, Unit):
            return ProductUnit(self, other)
        elif isinstance(other, (int, float)):
            return ScaledUnit(self, other)


class BaseUnit(Unit):
    label: Text

    def __init__(self, label: TextInput):
        self.label = Text.parse(label)

    def convert_to_base(self, value: float) -> float:
        return value

    def convert_from_base(self, base_value: float) -> float:
        return base_value

    def supports_conversion_to(self, other: "Unit") -> bool:
        return other == self

    @property
    def text(self) -> Text:
        return self.label


class ZeroUnit(BaseUnit):
    def __init__(self):
        super().__init__("0")

    def convert_to_base(self, value: float) -> float:
        return 0.0

    def convert_from_base(self, base_value: float) -> float:
        return 0.0

    @property
    def value_separator(self) -> Text:
        return lang.cdot

    def format_value(
        self, value: float, precision: Optional[int] = None, decimal: str = ","
    ) -> Text:
        return Text("0")

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, ZeroUnit)

    def __hash__(self) -> int:
        return hash(type(self))


class OneUnit(BaseUnit):
    def __init__(self):
        super().__init__("1")

    def convert_to_base(self, value: float) -> float:
        return value

    def convert_from_base(self, base_value: float) -> float:
        return base_value

    def format_value(
        self, value: float, precision: Optional[int] = None, decimal: str = ","
    ) -> Text:
        return lang.number(value, precision=precision, decimal=decimal)

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, OneUnit)

    def __hash__(self) -> int:
        return hash(type(self))


zero = ZeroUnit()
one = OneUnit()


class ScaledUnit(Unit):
    inner: Unit
    scale: float

    def __init__(self, base: Unit, scale: float):
        self.inner = base
        self.scale = float(scale)

    def convert_to_base(self, value: float) -> float:
        return self.inner.convert_to_base(value) * self.scale

    def convert_from_base(self, base_value: float) -> float:
        return self.inner.convert_from_base(base_value) / self.scale

    def supports_conversion_to(self, other: "Unit") -> bool:
        return self.inner.supports_conversion_to(other)

    @property
    def text(self) -> Text:
        if self.inner.text is None:
            return Text(str(self.scale))
        else:
            return str(self.scale) + lang.cdot + self.inner.text

    @property
    def dependencies(self) -> Iterable["Unit"]:
        return [self.inner]

    def simplify(self) -> "Unit":
        if self.scale == 0:
            return zero
        elif self.scale == 1:
            return self.inner
        else:
            return ScaledUnit(self.inner.simplify(), self.scale)

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, ScaledUnit)
            and self.inner == __o.inner
            and self.scale == __o.scale
        )

    def __hash__(self) -> int:
        return hash((type(self), self.inner, self.scale))


class OffsetUnit(Unit):
    inner: Unit
    offset: float

    def __init__(self, base: Unit, offset: float):
        self.inner = base
        self.offset = float(offset)

    def convert_to_base(self, value: float) -> float:
        return self.inner.convert_to_base(value) + self.offset

    def convert_from_base(self, base_value: float) -> float:
        return self.inner.convert_from_base(base_value) - self.offset

    def supports_conversion_to(self, other: "Unit") -> bool:
        return self.inner.supports_conversion_to(other)

    @property
    def base_unit(self) -> "Unit":
        return self.inner.base_unit

    @property
    def dependencies(self) -> Iterable["Unit"]:
        return [self.inner]

    @property
    def text(self) -> Text:
        if self.inner.text is None:
            return Text(str(self.offset))
        elif self.offset > 0:
            return self.inner.text + "+" + str(self.offset)
        else:
            return self.inner.text + "-" + str(-self.offset)

    def simplify(self) -> "Unit":
        if self.offset == 0:
            return self.inner.simplify()
        else:
            return OffsetUnit(self.inner.simplify(), self.offset)

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, OffsetUnit)
            and self.inner == __o.inner
            and self.offset == __o.offset
        )

    def __hash__(self) -> int:
        return hash((type(self), self.inner, self.offset))


class ProductUnit(Unit):
    factors: tuple[Unit, ...]
    separator: Text

    def __init__(self, *factors: Unit, separator: TextInput = lang.cdot):
        self.factors = factors
        self.separator = Text.parse(separator)

    def convert_to_base(self, value: float) -> float:
        base_value = value
        for factor in self.factors:
            base_value = factor.convert_to_base(base_value)
        return base_value

    def convert_from_base(self, base_value: float) -> float:
        value = base_value
        for factor in reversed(self.factors):
            value = factor.convert_from_base(value)
        return value

    def supports_conversion_to(self, other: "Unit") -> bool:
        return all(factor.supports_conversion_to(other) for factor in self.factors)

    @property
    def base_unit(self) -> "Unit":


    def extend(self, *factors: Unit):
        return ProductUnit(*(self.factors + factors), separator=self.separator)

    def simplify(self) -> "Unit":
        if len(self.factors) == 0:
            return one
        if len(self.factors) == 1:
            return self.factors[0]

        simplified_factors = []
        for factor in self.factors:
            simplified_factor = factor.simplify()
            if isinstance(simplified_factor, ProductUnit):
                simplified_factors.extend(simplified_factor.factors)
            elif simplified_factor == one:
                # Ignore one factors
                pass
            elif simplified_factor == zero:
                # If one factor is zero, the whole product is zero
                return zero
            else:
                simplified_factors.append(simplified_factor)
        return ProductUnit(*simplified_factors, separator=self.separator)

    @property
    def text(self) -> Text:
        return self.separator.join(factor.text for factor in self.factors)

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, ProductUnit) and self.factors == __o.factors

    def __hash__(self) -> int:
        return hash((type(self), self.factors))


class IntPowerUnit(Unit):
    base: Unit
    exp: int

    def __init__(self, base: Unit, exp: int):
        self.base = base
        self.exp = exp

    def _convert_to_base_with_exp(self, value: float, exp: int) -> float:
        if exp == 0:
            return 1.0
        elif exp > 0:
            for _ in range(exp):
                value = self.base.convert_to_base(value)
            return value
        else:
            for _ in range(-exp):
                value = self.base.convert_from_base(value)
            return value

    def convert_to_base(self, value: float) -> float:
        return self._convert_to_base_with_exp(value, self.exp)

    def convert_from_base(self, base_value: float) -> float:
        return self._convert_to_base_with_exp(base_value, -self.exp)

    @property
    def text(self) -> Optional[Text]:
        if self.base.text is None:
            return None
        return self.base.text.superscript(str(self.exp))

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, IntPowerUnit)
            and self.base == __o.base
            and self.exp == __o.exp
        )

    def __hash__(self) -> int:
        return hash((type(self), self.base, self.exp))


class DerivedUnit(Unit):
    label: Text
    unit: Unit

    def __init__(self, label: TextInput, unit: Unit):
        self.label = Text.parse(label)
        self.unit = unit

    def convert_to_base(self, value: float) -> float:
        return self.unit.convert_to_base(value)

    def convert_from_base(self, base_value: float) -> float:
        return self.unit.convert_from_base(base_value)

    @property
    def text(self) -> Text:
        return self.label
