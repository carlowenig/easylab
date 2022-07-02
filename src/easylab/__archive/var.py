from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from numbers import Number
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Optional,
    Protocol,
    SupportsFloat,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import sympy


T = TypeVar("T")

unset = "%%UNSET%%"

Maybe = Union[T, Literal["%%UNSET%%"]]


def fallback(maybe: Maybe[T], or_else: T) -> T:
    if maybe == "%%UNSET%%":
        return or_else
    else:
        return maybe


def merge_options(*options_tuple: dict[str, Any]):
    merged = {}
    for options in options_tuple:
        merged.update(options)
    return merged


class VarLike(Generic[T], metaclass=ABCMeta):
    type: Type[T]
    options: dict[str, Any]

    def __init__(self, type: Type[T] = Any, **options) -> None:
        super().__init__()
        self.type = type
        self.options = options

    @property
    @abstractmethod
    def expression(self) -> sympy.Expr:
        pass

    @property
    def dependencies(self) -> Tuple["VarLike", ...]:
        return ()

    def get_value(self, values: dict["Var", Any]) -> Optional[T]:
        return None

    @abstractmethod
    def copy(self, **options) -> "VarLike":
        pass

    def copy_deep(self, **options) -> "VarLike":
        return self.copy(**options)

    @property
    def root_dependencies(self) -> Tuple["VarLike", ...]:
        root_deps = []
        for dep in self.dependencies:
            inner_root_deps = dep.root_dependencies
            if len(inner_root_deps) == 0:
                root_deps.append(dep)
            else:
                root_deps.extend(dep.root_dependencies)
        return tuple(root_deps)

    def __add__(self, other):
        if not isinstance(other, VarLike):
            other = Const.from_value(other)
        return Computed(self.expression + other.expression, (self, other))

    def __radd__(self, other):
        if not isinstance(other, VarLike):
            other = Const.from_value(other)
        return Computed(other.expression + self.expression, (other, self))

    def __sub__(self, other):
        if not isinstance(other, VarLike):
            other = Const.from_value(other)
        return Computed(self.expression - other.expression, (self, other))

    def __rsub__(self, other):
        if not isinstance(other, VarLike):
            other = Const.from_value(other)
        return Computed(other.expression - self.expression, (other, self))

    def __mul__(self, other):
        if not isinstance(other, VarLike):
            other = Const.from_value(other)
        return Computed(self.expression * other.expression, (self, other))

    def __rmul__(self, other):
        if not isinstance(other, VarLike):
            other = Const.from_value(other)
        return Computed(other.expression * self.expression, (other, self))

    def __truediv__(self, other):
        if not isinstance(other, VarLike):
            other = Const.from_value(other)
        return Computed(self.expression / other.expression, (self, other))

    def __rtruediv__(self, other):
        if not isinstance(other, VarLike):
            other = Const.from_value(other)
        return Computed(other.expression / self.expression, (other, self))

    def __pow__(self, other):
        if not isinstance(other, VarLike):
            other = Const.from_value(other)
        return Computed(self.expression ** other.expression, (self, other))

    def __rpow__(self, other):
        if not isinstance(other, VarLike):
            other = Const.from_value(other)
        return Computed(other.expression ** self.expression, (other, self))

    # Specify __array__ method to allow ufuncs to work with VarLike objects
    def __array__(self, dtype=None):
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        sympy_args = []
        dependencies = []
        for input in inputs:
            if isinstance(input, VarLike):
                dependencies.append(input)
                sympy_args.append(input.expression)
            else:
                # Just use the input itself (e.g. for numbers)
                sympy_args.append(input)

        expr = getattr(sympy, ufunc.__name__)(*sympy_args)
        return Computed(expr, tuple(dependencies))

    def parse_value(self, input: Any) -> T:
        if "parser" in self.options:
            value = cast(T, self.options["parser"](input))
        elif hasattr(self.type, "__parse__"):
            kwargs = self.options.get("parse_kwargs", {})
            value = cast(T, getattr(self.type, "__parse__")(input, **kwargs))
        else:
            value = self.type(input)

        self.check_value(value)
        return value

    def check_value(self, input: T):
        if "checker" in self.options:
            self.options["checker"](input)

        if self.type != Any and not isinstance(input, self.type):
            raise TypeError(
                f"Got invalid value {input} for variable {self}. Expected type {self.type.__name__}, got {type(input).__name__}."
            )

    def format_value(self, value: T) -> str:
        if "formatter" in self.options:
            return self.options["formatter"](value)
        else:
            return str(value)

    def __str__(self) -> str:
        return str(self.expression)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}:\n"
        s += f"  expression: {self.expression}\n"
        if len(self.dependencies) > 0:
            s += f"  dependencies: {', '.join(str(d) for d in self.dependencies)}\n"
        return s.strip("\n")


class Var(VarLike[T]):
    _name: str

    def __init__(self, name: str, type: Type[T] = Any, **options):
        super().__init__(type, **options)
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def expression(self):
        return sympy.Symbol(self._name)

    def get_value(self, values: dict["VarLike", Any]) -> Optional[T]:
        return values[self]

    @property
    def dependencies(self):
        return ()

    def copy(self, *, name: Maybe[str] = unset, **options):
        return Var(
            fallback(name, self._name),
            **merge_options(self.options, options),
        )

    def sub(self, subscript: Any):
        subscript = str(subscript)
        if len(subscript) > 1:
            subscript = "{" + subscript + "}"
        return Var(self._name + "_" + subscript, **self.options)

    def to_const(self, value: T):
        return Const(self._name, value, **self.options)


class Const(VarLike[T]):
    _name: str
    _value: T

    def __init__(self, name: str, value: T, type: Type[T] = Any, **options):
        super().__init__(type, **options)
        self._name = name
        self.check_value(value)
        self._value = value

    @staticmethod
    def from_value(value: T):
        return Const(str(value), value)

    @property
    def expression(self):
        return sympy.Symbol(self._name)

    def get_value(self, values: dict["Var", Any]):
        return self._value

    @property
    def dependencies(self):
        return ()

    def copy(self, *, name: Maybe[str] = unset, value: Maybe[T] = unset, **options):
        return Const(
            fallback(name, self._name),
            fallback(value, self._value),
            **merge_options(self.options, options),
        )

    def to_var(self):
        return Var(self._name, **self.options)


class Computed(VarLike[T]):
    _expression: sympy.Expr
    _dependencies: Tuple["VarLike", ...]

    def __init__(
        self,
        expression: Union[sympy.Expr, "Any"],
        dependencies: Tuple["VarLike", ...] = (),
        type: Type[T] = Any,
        **options,
    ):
        super().__init__(type, **options)
        if isinstance(expression, Computed):
            if len(dependencies) > 0:
                raise ValueError(
                    "Cannot specify dependencies when passing a Computed object as first argument."
                )
            self._expression = expression._expression
            self._dependencies = expression._dependencies
        else:
            self._expression = expression
            self._dependencies = dependencies

    @property
    def expression(self):
        return self._expression

    @property
    def dependencies(self):
        return self._dependencies

    def get_value(self, values: dict["Var", Any]):
        root_deps = self.root_dependencies

        root_dep_values = []
        for dep in root_deps:
            value = dep.get_value(values)
            if value is None:
                return None
            root_dep_values.append(value)

        f = sympy.lambdify(
            [dep.expression for dep in root_deps], self._expression, "numpy"
        )

        value = f(*root_dep_values)
        self.check_value(value)
        return value

    def copy(
        self,
        *,
        expression: Maybe[sympy.Expr],
        dependencies: Maybe[Tuple["VarLike", ...]] = unset,
        **options,
    ):
        return Computed(
            fallback(expression, self._expression),
            fallback(dependencies, self._dependencies),
            **merge_options(self.options, options),
        )

    def copy_deep(self, *, expression: Maybe[sympy.Expr] = unset, **options):
        dependencies = tuple(d.copy_deep() for d in self._dependencies)
        return self.copy(
            expression=expression,
            dependencies=dependencies,
            **merge_options(self.options, options),
        )


# class Unit(VarLike[T]):
#     _name: str
#     _factor: float

#     def __init__(self, name: str, factor: float, type: Type[T] = Any, **options):
#         super().__init__(type, **options)
#         self._name = name
#         self._factor = factor

#     @property
#     def name(self):
#         return self._name

#     @property
#     def expression(self):
#         return sympy.Symbol(self._name)

#     def copy(
#         self, *, name: Maybe[str] = unset, factor: Maybe[float] = unset, **options
#     ):
#         return Unit(
#             fallback(name, self._name),
#             fallback(factor, self._factor),
#             **merge_options(self.options, options),
#         )


class Parsable(Protocol):
    @classmethod
    def __parse__(cls: Type[Self], input: Any, **kwargs) -> Self:
        ...


class Unit:
    """This class only serves as a type hint for unit variables."""

    def __init__(self) -> None:
        raise NotImplementedError()


class Units:
    s = Var("s", type=Unit)
    m = Var("m", type=Unit)

    g = Var("g", type=Unit)
    kg = 1e3 * g


class Measured(Parsable, np.lib.mixins.NDArrayOperatorsMixin):
    value: float
    error: float = 0
    unit: Optional[VarLike] = None
    prec: Optional[int] = None

    def __init__(
        self,
        value: float,
        *,
        error: float = 0,
        unit: Optional[VarLike] = None,
        prec: Optional[int] = None,
    ):
        self.value = value
        self.error = error
        self.unit = unit
        self.prec = prec

    @classmethod
    def __parse__(cls: Type[Self], input: Any, **kwargs) -> Self:
        if isinstance(input, cls):
            return input
        elif isinstance(input, SupportsFloat):
            return cls(float(input), **kwargs)
        elif isinstance(input, tuple) and len(input) == 2:
            return cls(input[0], error=input[1], **kwargs)
        else:
            raise ValueError(f"Cannot parse {input} as {cls.__name__}.")

    def __str__(self):
        s = f"{self.value:.3g}"
        if self.error != 0:
            s += f" ± {self.error:.3g}"
        if self.unit is not None:
            if self.error != 0:
                s = f"({s})"
            s += f" {self.unit}"
        return s


class State:
    _values: dict[Var, Any]

    def __init__(self, values: dict[Var, Any]):
        self._values = {}
        for var, value in values.items():
            if not isinstance(var, Var):
                raise ValueError(
                    f"State should only contain Var objects as keys. Got {type(var)}."
                )
            self._values[var] = var.parse_value(value)

    @property
    def vars(self):
        return tuple(self._values.keys())

    @property
    def values(self):
        return self._values

    def __getitem__(self, var: VarLike[T]) -> T:
        value = var.get_value(self._values)
        if value is None:
            raise ValueError(f"{self} does not contain a value for {var}.")
        return value

    def __str__(self):
        return f"State({self._values})"

    __repr__ = __str__


class Series(Generic[T]):
    _var: Var[T]
    _values: np.ndarray

    def __init__(self, var: Var[T], values: Iterable[T]):
        self._var = var
        self._values = np.array(values)

    @property
    def var(self):
        return self._var

    @property
    def values(self):
        return self._values

    def __getitem__(self, index: int) -> T:
        return self._values[index]

    def __str__(self):
        return f"Series({self._var}, {self._values})"

    __repr__ = __str__


class Table:
    _vars: list[Var]
    _df: pd.DataFrame

    def __init__(self, vars: Iterable[Var] = (), data: Any = None):
        self._vars = list(vars)
        self._df = pd.DataFrame(data)

    @staticmethod
    def from_states(states: Iterable[Union[State, dict[Var, Any]]]):
        vars = None
        data = pd.DataFrame()
        for state in states:
            if not isinstance(state, State):
                state = State(state)

            if vars is None:
                vars = state.vars
            elif vars != state.vars:
                raise ValueError("All states must have the same variables.")

            data = data.append(
                {var.name: value for var, value in state.values.items()},
                ignore_index=True,
            )

        if vars is None:
            return Table()

        return Table(vars, data)

    @overload
    def __getitem__(self, selector: Var[T]) -> Series[T]:
        ...

    @overload
    def __getitem__(self, selector: int) -> State:
        ...

    def __getitem__(self, selector: Union[Var[T], int]) -> Union[Series[T], State]:
        if isinstance(selector, int):
            row = self._df.iloc[selector]
            return State({var: row[var.name] for var in self._vars})
        elif isinstance(selector, Var):
            return Series(selector, self._df[selector.name])
        else:
            raise ValueError(f"Invalid selector: {selector}")

    def __str__(self) -> str:
        return str(self._df)


x = Var("x", type=Measured, parse_kwargs={"unit": Units.m})
y = Var("y", type=float, formatter=lambda value: f"{value:.3f}")

x_sq = x ** 2
print(x_sq)

a = Computed(np.cos(x + y))
print(a)

table = Table.from_states(
    [
        {x: 1, y: 2.5},
        {x: (3, 0.5), y: 4},
    ]
)

print(table)
