from __future__ import annotations
from abc import ABC, abstractmethod
import math
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
    overload,
    runtime_checkable,
)
from typing_extensions import Self
import pandas as pd
import numpy as np
from varname import varname

from ..lang import ascii, text, Text, lang

_T = TypeVar("_T")


_last_labeled_indices: dict[type, int] = {}


class Labeled:
    __label: Text

    def __init__(self, label: Any = None):
        self.label = label  # Use setter to ensure that label is a Text object.

    @property
    def label(self) -> Text:
        return self.__label

    @label.setter
    def label(self, label: Any):
        if label is None:
            t = type(self)
            if t not in _last_labeled_indices:
                _last_labeled_indices[t] = 0
            self.__label = Text(t.__name__).subscript(_last_labeled_indices[t])
        else:
            self.__label = text(label)

    def __str__(self) -> str:
        return ascii(self.__label)


type_defaults: dict[Any, Any] = {
    Any: None,
    type(None): None,
    bool: False,
    int: 0,
    float: math.nan,
    str: "",
}


def get_type_default(type_: type[_T] | Any) -> _T:
    if type_ in type_defaults:
        return cast(_T, type_defaults[type_])

    if issubclass(type(None), type_):
        return cast(_T, None)
    try:
        return type_()
    except:
        raise ValueError(f"Cannot get default value for type {type_}.")


class Undefined:
    def __new__(cls: type[Self]) -> Self:
        return undefined

    def __init_subclass__(cls) -> None:
        raise TypeError("Cannot subclass NotGiven.")


undefined = object.__new__(Undefined)


class Expr(Labeled, Generic[_T]):
    def __init__(self, label: Any, dependencies: tuple[Expr[Any], ...] = ()):
        self.dependencies = dependencies
        super().__init__(label)

    def _evaluate(self, data: Data) -> _T:
        raise NotImplementedError(f"Cannot evaluate {type(self).__name__}.")

    def evaluate(self, data: Data) -> _T:
        return self._evaluate(data)

    def _differentiate(self, dep: Expr[Any]) -> Expr[_T]:
        raise NotImplementedError(f"Cannot differentiate {type(self).__name__}.")

    def differentiate(self, dep: Expr[Any]) -> Expr[_T]:
        if dep not in self.dependencies:
            raise ValueError(
                f"Cannot differentiate {self} with respect to {dep}, since it not a dependency."
            )
        return self._differentiate(dep)

    def simplify(self) -> Expr[_T]:
        return self

    def __eq__(self, other: Any) -> bool:
        diff = self - other
        if isinstance(diff, Expr):
            diff = diff.simplify()
        return is_zero(diff)


def zero(value_: _T) -> _T:
    if isinstance(value_, int):
        result = 0
    elif isinstance(value_, float):
        result = 0.0
    elif isinstance(value_, complex):
        result = 0j
    elif isinstance(value_, str):
        result = "0"
    elif isinstance(value_, bool):
        result = False
    elif isinstance(value_, np.ndarray):
        result = np.zeros_like(value_)
    elif isinstance(value_, pd.Series):
        result = pd.Series(0, index=value_.index)
    elif isinstance(value_, pd.DataFrame):
        result = pd.DataFrame(0, index=value_.index, columns=value_.columns)
    elif hasattr(value_, "__zero__"):
        result = getattr(value_, "__zero__")()
    else:
        raise ValueError(f"Cannot get zero for {value_}.")
    return cast(_T, result)


def one(value_: _T) -> _T:
    if isinstance(value_, int):
        result = 1
    elif isinstance(value_, float):
        result = 1.0
    elif isinstance(value_, complex):
        result = 1j
    elif isinstance(value_, str):
        result = "1"
    elif isinstance(value_, bool):
        result = True
    elif isinstance(value_, np.ndarray):
        result = np.ones_like(value_)
    elif isinstance(value_, pd.Series):
        result = pd.Series(1, index=value_.index)
    elif isinstance(value_, pd.DataFrame):
        result = pd.DataFrame(1, index=value_.index, columns=value_.columns)
    elif hasattr(value_, "__one__"):
        result = getattr(value_, "__one__")()
    else:
        raise ValueError(f"Cannot get one for {value_}.")
    return cast(_T, result)


def is_zero(value: Any):
    return value + value == value


def is_one(value: Any):
    return value * value == value


class Constant(Expr[_T]):
    def __init__(self, value: _T, label: Any = None):
        self.value = value
        super().__init__(label or value)

    def _evaluate(self, data: Data) -> _T:
        return self.value

    def _differentiate(self, dep: Expr[Any]) -> Expr[_T]:
        return Constant(zero(self.value))


class SumExpr(Expr[_T]):
    def __init__(self, a: Expr[_T], b: Expr[_T]):
        self.a = a
        self.b = b
        super().__init__(a.label + " + " + b.label, (a, b))

    def _evaluate(self, data: Data) -> _T:
        return self.a.evaluate(data) + self.b.evaluate(data)  # type: ignore

    def _differentiate(self, dep: Expr[Any]) -> Expr[_T]:
        return SumExpr(self.a.differentiate(dep), self.b.differentiate(dep)).simplify()

    def simplify(self) -> Expr[_T]:
        a = self.a.simplify()
        b = self.b.simplify()
        if isinstance(a, Constant) and isinstance(b, Constant):
            return Constant(a.value + b.value)  # type: ignore
        elif is_zero(a):
            return b
        elif is_zero(b):
            return a
        return SumExpr(a, b)


class ProductExpr(Expr[_T]):
    def __init__(self, a: Expr[_T], b: Expr[_T]):
        self.a = a
        self.b = b
        super().__init__(a.label * b.label, (a, b))


class FunctionExpr(Expr[_T]):
    def __init__(
        self,
        label: Any,
        call: Callable[..., _T],
    ):
        self.call = call
        super().__init__(label)

    def __call__(self, *args: Expr[Any]) -> Expr[_T]:
        return EvaluatedFunctionExpr(self, args)


class EvaluatedFunctionExpr(Expr[_T]):
    def __init__(self, func: FunctionExpr[_T], args: tuple[Expr[_T], ...]):
        self.func = func
        self.args = args
        super().__init__(lang.mathrm(func.label) + text(args), (func, *args))

    def _evaluate(self, data: Data) -> _T:
        return self.func.call(*(arg.evaluate(data) for arg in self.args))

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, EvaluatedFunctionExpr)
            and self.func == other.func
            and self.args == other.args
        )

    def __hash__(self) -> int:
        return hash((self.func, *self.args))


sin = FunctionExpr("sin", np.sin)
cos = FunctionExpr("cos", np.cos)
tan = FunctionExpr("tan", np.tan)
arcsin = FunctionExpr("arcsin", np.arcsin)
arccos = FunctionExpr("arccos", np.arccos)
arctan = FunctionExpr("arctan", np.arctan)


class Var(Generic[_T], Labeled):
    def __init__(
        self,
        label: Any = None,
        type_: type[_T] | None = None,
        default: _T | Undefined = undefined,
    ):
        if (
            type_ is not None
            and default is not undefined
            and not isinstance(default, type_)
        ):
            raise TypeError(
                f"Default value {default} is not of specified type {type_.__name__}."
            )
        elif default is undefined:
            default = get_type_default(type_)
        elif type_ is None:
            type_ = type(default)

        self.__type = cast(type[_T], type_)
        self.__default = cast(_T, default)
        super().__init__(label)

    @property
    def type(self):
        return self.__type

    @property
    def default(self):
        return self.__default

    def copy(self):
        return Var(self.label, self.__type, self.__default)

    def __repr__(self):
        return f"Var(label={repr(self.label)}, type={self.__type}, default={self.__default})"


VarIdentifier = Union[Var[_T], Any]


class Data(Labeled):
    @property
    def vars(self) -> Iterable[Var]:
        return []

    def get_var(self, var: VarIdentifier[_T], /) -> Var[_T]:
        vars = self.vars
        if isinstance(var, Var):
            if var in vars:
                return var
            else:
                raise ValueError(f"Variable {var} not found in {self}.")
        else:
            for var_ in vars:
                if var_.label.matches(var):
                    return var_
            raise ValueError(f"Variable {var} not found in {self}.")

    def get(self, var: VarIdentifier[_T]) -> _T:
        var = self.get_var(var)
        return var.default

    @abstractmethod
    def set(self, var: VarIdentifier[_T], value: _T) -> None:
        pass

    @abstractmethod
    def copy(self) -> Self:
        pass

    @abstractmethod
    def to_data_frame(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def __repr__(self):
        pass


class DictBasedData(Data):
    def __init__(self, __dict: dict[Var, Any], label: Any = None) -> None:
        self.__dict = __dict
        super().__init__(label)

    @property
    def vars(self) -> Iterable[Var]:
        return self.__dict.keys()

    def get_var(self, var: VarIdentifier[_T]) -> Var[_T]:
        if isinstance(var, Var):
            return var
        for v in self.vars:
            if v.label == var:
                return v
        raise ValueError(f"Var {var} not found.")

    def get(self, var: VarIdentifier[_T]) -> _T:
        var = self.get_var(var)
        return self.__dict[var]

    def set(self, var: VarIdentifier[_T], value: _T) -> None:
        var = self.get_var(var)
        self.__dict[var] = value

    def copy(self) -> Self:
        return DictBasedData(self.__dict.copy())

    def to_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.__dict)

    def __repr__(self):
        return f"{type(self).__name__}({self.__dict!r})"


__all__ = [
    "Labeled",
    "type_defaults",
    "get_type_default",
    "Var",
    "Data",
    "DictBasedData",
]
