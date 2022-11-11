from __future__ import annotations
from abc import ABC, abstractmethod
import inspect
import keyword
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    overload,
)

from ..lang import Text
from ..util import undefined, Undefined, Wildcard, is_wildcard

from . import record
from .var_type import VarType, VarTypeLike

T = TypeVar("T")


def get_neutral(type_: type[T]) -> T:
    if issubclass(int, type_):
        return cast(T, 0)
    elif issubclass(float, type_):
        return cast(T, 0.0)
    elif issubclass(str, type_):
        return cast(T, "")
    elif issubclass(bool, type_):
        return cast(T, False)
    elif issubclass(list, type_):
        return cast(T, [])
    elif issubclass(dict, type_):
        return cast(T, {})
    elif issubclass(tuple, type_):
        return cast(T, ())
    elif issubclass(set, type_):
        return cast(T, set())
    elif issubclass(Text, type_):
        return cast(T, Text(""))
    elif issubclass(Iterable, type_):
        return cast(T, [])
    elif issubclass(Callable, type_):
        return cast(T, lambda: None)
    elif issubclass(type, type_):
        return cast(T, type(None))
    elif hasattr(type_, "__neutral__"):
        __neutral__ = getattr(type_, "__neutral__")
        if callable(__neutral__):
            return __neutral__()
        else:
            return __neutral__
    else:
        raise TypeError(f"Cannot get neutral element for type {type_!r}.")


class Var(Generic[T]):
    def __init__(
        self,
        label: Any,
        type_: VarTypeLike[T] = object,
    ) -> None:
        self._label = Text.interpret(label)
        self._type = VarType.interpret(type_)

    @property
    def label(self) -> Text:
        return self._label

    @label.setter
    def label(self, label: Any) -> None:
        self._label = Text.interpret(label)

    @property
    def type(self) -> VarType[T]:
        return self._type

    @type.setter
    def type(self, type_: VarTypeLike[T]) -> None:
        self._type = VarType.interpret(type_)

    def default(self) -> T | Undefined:
        return self.type.default()

    def has_default(self):
        return self.default() is not undefined

    def __repr__(self) -> str:
        return f"{type(self).__name__}(label={self.label!r}, type={self.type!r})"

    def __str__(self) -> str:
        return self.label.ascii

    @overload
    def format(self, value: Any, *, check: bool = True) -> Text:
        ...

    @overload
    def format(self, value: T, *, parse: Literal[False], check: bool = True) -> Text:
        ...

    def format(self, value: Any, *, parse: bool = True, check: bool = True) -> Text:
        if parse:
            value = self.type.parse(value)

        if check:
            self.type.check(value)

        return self.type.format(value)

    def parse(self, raw: Any, *, check: bool = True) -> T:
        result = self.type.parse(raw)
        if check:
            self.type.check(result)
        return result

    @overload
    def equal(self, a: Any, b: Any, *, check: bool = True) -> bool:
        ...

    @overload
    def equal(self, a: T, b: T, *, parse: Literal[False], check: bool = True) -> bool:
        ...

    def equal(self, a: Any, b: Any, *, parse: bool = True, check: bool = True) -> bool:
        if parse:
            a = self.type.parse(a)
            b = self.type.parse(b)

        if check:
            self.type.check(a)
            self.type.check(b)

        return self.type.equal(a, b)

    @overload
    def check(self, value: Any, *, parse: bool = True) -> None:
        ...

    @overload
    def check(self, value: T, *, parse: Literal[False]) -> None:
        ...

    def check(self, value: Any, *, parse: bool = True) -> None:
        if parse:
            value = self.type.parse(value)

        self.type.check(value)

    def entry(self, value: Any):
        from ..data import RecordEntry

        return RecordEntry(self, self.parse(value))

    def matches(self, query: VarQuery) -> bool:
        return is_wildcard(query) or query is self or self.label.matches(query)

    def __eq__(self, other: Any) -> VarCondition:
        return VarCondition.for_other(self, other, lambda x, y: x == y)

    def __ne__(self, other: Any) -> VarCondition:
        return VarCondition.for_other(self, other, lambda x, y: x != y)

    def __lt__(self, other: Any) -> VarCondition:
        return VarCondition.for_other(self, other, lambda x, y: x < y)

    def __le__(self, other: Any) -> VarCondition:
        return VarCondition.for_other(self, other, lambda x, y: x <= y)

    def __gt__(self, other: Any) -> VarCondition:
        return VarCondition.for_other(self, other, lambda x, y: x > y)

    def __ge__(self, other: Any) -> VarCondition:
        return VarCondition.for_other(self, other, lambda x, y: x >= y)

    def __contains__(self, other: Any) -> VarCondition:
        return VarCondition.for_other(self, other, lambda x, y: y in x)

    def __hash__(self):
        return id(self)

    def sub(self, *subscripts: Any):
        return Var(self.label.subscript(Text(", ").join(subscripts)), self.type)


VarQuery = Union[Var[T], Wildcard, Any]


class VarCondition:
    @staticmethod
    def for_other(
        var: Var, other: Any, check: Callable[[Any, Any], bool]
    ) -> VarCondition:
        if isinstance(other, Var):
            return VarCondition([var, other], lambda x, y: check(x, y))
        else:
            return VarCondition([var], lambda x: check(x, other))

    def __init__(self, vars: Iterable[Var[Any]], check: Callable[..., bool]) -> None:
        self.vars = list(vars)
        self.check = check

        for var in self.vars:
            if not isinstance(var, Var):
                raise TypeError(
                    f"First argument of VarCondition must be a list of Var objects. Found element of type {type(var).__name__}."
                )

    def __and__(self, other: VarCondition):
        n_vars = len(self.vars)
        return VarCondition(
            self.vars + other.vars,
            lambda *args: self.check(*args[:n_vars]) and other.check(*args[n_vars:]),
        )

    def __or__(self, other: VarCondition):
        n_vars = len(self.vars)
        return VarCondition(
            self.vars + other.vars,
            lambda *args: self.check(*args[:n_vars]) or other.check(*args[n_vars:]),
        )

    def __invert__(self):
        return VarCondition(self.vars, lambda *args: not self.check(*args))


class DerivedVar(Var[T], ABC):
    @abstractmethod
    def get_value(self, record: "record.Record") -> T:
        ...


class Const(DerivedVar[T]):
    def __init__(self, label: Any, value: T, type_: type[T] | None = None) -> None:
        if type_ is None:
            type_ = type(value)
        elif not isinstance(value, type_):
            raise TypeError(f"Value {value!r} is not of specified type {type_!r}.")

        self.value = value

        super().__init__(label, type_)

    def get_value(self, record: "record.Record") -> T:
        return self.value


def infer_return_type(
    f: Callable[..., T], param_types: list[type] | None = None
) -> type[T]:
    sig = inspect.signature(f, eval_str=True)

    if (
        isinstance(sig.return_annotation, type)
        and sig.return_annotation is not sig.empty
    ):
        return cast(type[T], sig.return_annotation)

    if param_types is None:
        param_types = []

        for param in sig.parameters.values():
            if param.annotation is not param.empty and isinstance(
                param.annotation, type
            ):
                param_types.append(param.annotation)
            elif param.default is not param.empty:
                param_types.append(type(param.default))
            else:
                param_types = None
                break

    fail_reason = None

    if param_types is not None:
        test_args = []
        test_args_available = True

        for param_type in param_types:
            try:
                test_args.append(get_neutral(param_type))
            except TypeError:
                try:
                    test_args.append(param_type())
                except:
                    test_args_available = False
                    fail_reason = f"Could not create test argument for type {param_type.__name__}."
                    break

        if test_args_available:
            try:
                return type(f(*test_args))
            except Exception as e:
                fail_reason = e

    raise TypeError(
        f"Cannot infer return type of {f!r}. Please specify it explicitly."
        + (f" Reason: {fail_reason!r}" if fail_reason else "")
    )


def sanitize_arg_name(name: str):
    s = "".join(
        c if c.isalnum() else "_" for c in name.replace(" ", "_").replace("-", "_")
    )

    if s[0].isdigit():
        s = "_" + s

    if keyword.iskeyword(s):
        s += "_"

    if not s.isidentifier():
        raise ValueError(f"Invalid argument name {name!r}.")

    return s


class Computed(DerivedVar[T]):
    def __init__(
        self,
        label: Any,
        params: Iterable[Var],
        compute: Callable[..., Any] | str | type[T] | None = None,
        type_: type[T] | None = None,
    ) -> None:
        if isinstance(compute, type):
            type_ = cast(type[T], compute)
            compute = None

        if compute is None:
            compute = Text.interpret(label).ascii

        if isinstance(compute, str):
            compute = cast(
                Callable[..., Any],
                eval(
                    f"lambda {', '.join(sanitize_arg_name(param.label.ascii) for param in params)}: {compute}",
                ),
            )

        assert callable(compute)

        if type_ is None:
            type_ = infer_return_type(
                compute, [param.type.value_type for param in params]
            )

        assert isinstance(type_, type)

        self._params = tuple(params)
        self.compute = compute

        super().__init__(label, type_)

    @property
    def params(self) -> tuple[Var, ...]:
        return self._params

    @params.setter
    def params(self, params: Iterable[Var]) -> None:
        self._params = tuple(params)

    def __call__(self, *args: Any) -> T:
        if len(args) != len(self.params):
            raise TypeError(f"Expected {len(self.params)} arguments, got {len(args)}.")

        parsed_args = [param.parse(arg) for param, arg in zip(self.params, args)]

        result = self.compute(*parsed_args)

        return self.parse(result)

    def get_value(self, record: "record.Record") -> T:
        return self(*[record[param] for param in self.params])
