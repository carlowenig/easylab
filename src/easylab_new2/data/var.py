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

from ..lang import Text, lang
from ..util import undefined, Undefined, Wildcard, is_wildcard

from . import record
from .var_type import VarType, VarTypeLike, is_var_type_like
from .metadata import Metadata, MetadataLike

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


def interpret_var_label(
    label: Any,
    math: bool = True,
    smart_subscript: bool = True,
    smart_superscript: bool = True,
) -> Text:
    if isinstance(label, Text):
        return label
    else:
        text = Text(label)

        if smart_subscript and "_" in text.ascii:
            parts = text.ascii.split("_")
            text = Text(parts[0])
            for subscript in parts[1:]:
                text = text.subscript(subscript)

        if smart_superscript and "^" in text.ascii:
            parts = text.ascii.split("^")
            text = Text(parts[0])
            for superscript in parts[1:]:
                text = text.superscript(superscript)

        if math:
            text = lang.math(text)

        return text


class Var(Generic[T]):
    def __init__(
        self,
        label: Any,
        type_: VarTypeLike[T] = object,
        metadata: MetadataLike = None,
    ) -> None:
        self._label = interpret_var_label(label)
        self._type = VarType.interpret(type_)
        self.metadata = Metadata.interpret(metadata)

    @property
    def label(self) -> Text:
        return self._label

    @label.setter
    def label(self, label: Any) -> None:
        self._label = Text.interpret(label)
        self.metadata.update()

    @property
    def type(self) -> VarType[T]:
        return self._type

    @type.setter
    def type(self, type_: VarTypeLike[T]) -> None:
        self._type = VarType.interpret(type_)
        self.metadata.update()

    def default(self) -> T | Undefined:
        return self.type.default()

    def has_default(self):
        return self.default() is not undefined

    def __repr__(self) -> str:
        return f"{type(self).__name__}(label={self.label!r}, type={self.type!r}, metadata={self.metadata!r})"

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

    def compute_by(
        self,
        params: tuple[Var, ...],
        func: Callable[..., Any] | str,
    ) -> Callable[..., T]:
        return self.type.compute_by(params, func)

    def get_plot_value(
        self, value: Any, *, parse: bool = True, check: bool = True
    ) -> Any:
        if parse:
            value = self.type.parse(value)

        if check:
            self.type.check(value)

        return self.type.get_plot_value(value)

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
        if is_wildcard(query) or query is self:
            return True

        if isinstance(query, str):
            if self.label.matches(query):
                return True

            if ":" in query:
                key, value = query.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key == "label":
                    return self.label.matches(value)
                if key == "type":
                    return self.type.matches(value)
                if key == "metadata":
                    return self.metadata.matches(value)

            return False

        if isinstance(query, dict):
            if "label" in query and not self.label.matches(query["label"]):
                return False
            if "type" in query and not self.type.matches(query["type"]):
                return False
            if "metadata" in query and not self.metadata.matches(query["metadata"]):
                return False

            metadata_dict = {
                key.removeprefix("metadata."): value
                for key, value in query.items()
                if key.startswith("metadata.")
            }
            if metadata_dict and not self.metadata.matches(metadata_dict):
                return False

            return True

        if isinstance(query, Var):
            return self.label.matches(query.label)

        return False
        # raise TypeError(f"Cannot match var {self} by query of type {type(query).__name__}.")

    def __eq__(self, other: Any) -> RecordEntryCondition:
        return RecordEntryCondition.for_other(self, other, lambda x, y: x == y)

    def __ne__(self, other: Any) -> RecordEntryCondition:
        return RecordEntryCondition.for_other(self, other, lambda x, y: x != y)

    def __lt__(self, other: Any) -> RecordEntryCondition:
        return RecordEntryCondition.for_other(self, other, lambda x, y: x < y)

    def __le__(self, other: Any) -> RecordEntryCondition:
        return RecordEntryCondition.for_other(self, other, lambda x, y: x <= y)

    def __gt__(self, other: Any) -> RecordEntryCondition:
        return RecordEntryCondition.for_other(self, other, lambda x, y: x > y)

    def __ge__(self, other: Any) -> RecordEntryCondition:
        return RecordEntryCondition.for_other(self, other, lambda x, y: x >= y)

    def __contains__(self, other: Any) -> RecordEntryCondition:
        return RecordEntryCondition.for_other(self, other, lambda x, y: y in x)

    def __hash__(self):
        return id(self)

    def sub(self, *subscripts: Any):
        return Var(self.label.subscript(Text(", ").join(subscripts)), self.type)


VarQuery = Union[Var[T], Wildcard, Any]


class RecordEntryCondition:
    @staticmethod
    def for_other(
        var: Var, other: Any, check: Callable[[Any, Any], bool]
    ) -> RecordEntryCondition:
        if isinstance(other, Var):
            return RecordEntryCondition([var, other], lambda x, y: check(x, y))
        else:
            return RecordEntryCondition([var], lambda x: check(x, other))

    def __init__(self, vars: Iterable[Var[Any]], check: Callable[..., bool]) -> None:
        self.vars = list(vars)
        self.check = check

        for var in self.vars:
            if not isinstance(var, Var):
                raise TypeError(
                    f"First argument of VarCondition must be a list of Var objects. Found element of type {type(var).__name__}."
                )

    def __and__(self, other: RecordEntryCondition):
        n_vars = len(self.vars)
        return RecordEntryCondition(
            self.vars + other.vars,
            lambda *args: self.check(*args[:n_vars]) and other.check(*args[n_vars:]),
        )

    def __or__(self, other: RecordEntryCondition):
        n_vars = len(self.vars)
        return RecordEntryCondition(
            self.vars + other.vars,
            lambda *args: self.check(*args[:n_vars]) or other.check(*args[n_vars:]),
        )

    def __invert__(self):
        return RecordEntryCondition(self.vars, lambda *args: not self.check(*args))


class DerivedVar(Var[T], ABC):
    @abstractmethod
    def get_value(self, record: "record.Record") -> T:
        ...

    @abstractmethod
    def get_dependencies(self) -> Iterable[Var]:
        ...


class Const(DerivedVar[T]):
    def __init__(
        self, label: Any, value: T | Any, type_: VarTypeLike[T] | None = None
    ) -> None:
        super().__init__(label, type_ if type_ is not None else type(value))

        self._value = self.parse(value)

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        self._value = self.parse(value)
        self.metadata.update()

    def get_value(self, record: "record.Record") -> T:
        return self._value

    def get_dependencies(self) -> Iterable[Var]:
        return []


def infer_return_type(
    f: Callable[..., T] | str, param_types: list[type] | None = None
) -> type[T]:
    fail_reason = None

    if callable(f):
        sig = inspect.signature(f)  # , eval_str=True)

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
    else:
        fail_reason = "Object is not callable."

    raise TypeError(
        f"Cannot infer return type of {f!r}. Please specify it explicitly."
        + (f" Reason: {fail_reason!r}" if fail_reason else "")
    )


class Computed(DerivedVar[T]):
    def __init__(
        self,
        label: Any,
        params: Iterable[Var],
        func: Callable[..., Any] | str,
        type_: VarTypeLike[T] | None = None,
    ) -> None:
        self._params = tuple(params)

        if type_ is None:
            type_ = infer_return_type(func, [param.type.value_type for param in params])

        type_ = VarType.interpret(cast(VarTypeLike, type_))

        super().__init__(label, type_)

        self._func = func
        self._compute = self.compute_by(self._params, func)

    @property
    def params(self) -> tuple[Var, ...]:
        return self._params

    @params.setter
    def params(self, params: Iterable[Var]) -> None:
        self._params = tuple(params)

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, func: Callable[..., Any] | str) -> None:
        self._func = func
        self._compute = self.compute_by(self._params, func)

    def compute(self, *param_vals: Any, parse: bool = True, check: bool = True) -> T:
        result = self._compute(*param_vals)

        if parse:
            result = self.parse(result, check=check)
        elif check:
            self.check(result)

        return cast(T, result)

    @property
    def type(self) -> VarType[T]:
        return super().type

    @type.setter
    def type(self, type_: VarTypeLike[T]) -> None:
        super().type = type_
        self.func = self._func  # Trigger func setter

    def __call__(self, *args: Any) -> T:
        if len(args) != len(self.params):
            raise TypeError(f"Expected {len(self.params)} arguments, got {len(args)}.")

        parsed_args = [param.parse(arg) for param, arg in zip(self.params, args)]

        result = self.compute(*parsed_args)

        return self.parse(result)

    def get_value(self, record: "record.Record") -> T:
        return self(*[record[param] for param in self.params])

    def get_dependencies(self) -> Iterable[Var]:
        return self.params
