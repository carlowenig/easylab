from __future__ import annotations
from abc import ABC, abstractmethod
import inspect
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import TypeGuard

from ..lang import Text, lang
from ..internal_util import undefined, Undefined, Wildcard, is_wildcard
from ..expr import Expr, ExprLike, Symbol

from . import record as m_record
from .var_type import VarType, VarTypeLike
from .metadata import Metadata, MetadataLike

T = TypeVar("T")
S = TypeVar("S")


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


# import sympy


# VarExprLike = Union["VarExpr[T]", "Var[T]", tuple[Iterable["Var"], Any], Any]


# def get_var_type_from_sympy(expr: sympy.Expr) -> VarType | None:
#     if isinstance(expr, sympy.Integer):
#         return VarType.interpret(int)
#     elif isinstance(expr, sympy.Float):
#         return VarType.interpret(float)


# class VarExpr(Generic[T]):
#     @staticmethod
#     def interpret(input: VarExprLike[T]) -> VarExpr[T]:
#         if isinstance(input, VarExpr):
#             return input
#         elif isinstance(input, Var):
#             return VarExpr(sympy.Symbol(input.label.latex), input.type, [input])
#         elif isinstance(input, tuple) and len(input) == 2:
#             deps, expr = input
#             expr: sympy.Expr = sympy.sympify(input)
#             type_ = get_var_type_from_sympy(expr)
#             return VarExpr(expr, type_ or cast(type[T], object), deps)
#         else:
#             return VarExpr(input, cast(type[T], object), [])

#     def __init__(
#         self, sympy_expr: Any, type_: VarTypeLike[T], dependencies: Iterable[Var]
#     ) -> None:
#         self._sympy_expr: sympy.Expr = sympy.sympify(sympy_expr)
#         self._type = VarType.interpret(type_)
#         self._dependencies = list(dependencies)

#     @property
#     def sympy_expr(self):
#         return self._sympy_expr

#     @property
#     def type(self) -> VarType[T]:
#         return self._type

#     @property
#     def dependencies(self) -> list[Var]:
#         return self._dependencies

#     @functools.cache
#     def _create_evaluator(self, **lambdify_kwargs) -> Callable:
#         return sympy.lambdify(
#             [dep.sympy_expr for dep in self.dependencies],
#             self.sympy_expr,
#             **lambdify_kwargs,
#         )

#     def evaluate_for_args(self, *args) -> T:
#         evaluator = self._create_evaluator()
#         result = self.type.parse(evaluator(*args))
#         self.type.check(result)
#         return result

#     def evaluate_for_record(self, record: "m_record.Record") -> T:
#         args = [record[dep] for dep in self.dependencies]
#         return self.evaluate_for_args(*args)

#     def transform(self, f: Callable[[sympy.Expr], Any]):
#         return VarExpr(f(self.sympy_expr), self.type, self.dependencies)

#     def combine(self, other: VarExprLike, f: Callable[[sympy.Expr, sympy.Expr], Any]):
#         other = VarExpr.interpret(other)
#         return VarExpr(
#             f(self.sympy_expr, other.sympy_expr),
#             self.type,
#             self.dependencies + other.dependencies,
#         )

#     @property
#     @functools.cache
#     def text(self):
#         latex = sympy.latex(self.sympy_expr, mode="inline")
#         return Text(str(self.sympy_expr), latex=latex)

#     def __repr__(self):
#         return self.text.ascii

#     def __str__(self):
#         return self.text.ascii

#     def __add__(self, other: VarExprLike):
#         return self.combine(other, lambda a, b: a + b)

#     def __radd__(self, other: VarExprLike):
#         return VarExpr.interpret(other) + self

#     def __sub__(self, other: VarExprLike):
#         return self.combine(other, lambda a, b: a - b)

#     def __rsub__(self, other: VarExprLike):
#         return VarExpr.interpret(other) - self

#     def __mul__(self, other: VarExprLike):
#         return self.combine(other, lambda a, b: a * b)

#     def __rmul__(self, other: VarExprLike):
#         return VarExpr.interpret(other) * self

#     def __truediv__(self, other: VarExprLike):
#         return self.combine(other, lambda a, b: a / b)

#     def __rtruediv__(self, other: VarExprLike):
#         return VarExpr.interpret(other) / self

#     def __pow__(self, other: VarExprLike):
#         return self.combine(other, lambda a, b: a ** b)

#     def __rpow__(self, other: VarExprLike):
#         return VarExpr.interpret(other) ** self

#     def __neg__(self):
#         return self.transform(lambda a: -a)

#     def __pos__(self):
#         return self

#     def __abs__(self):
#         return self.transform(abs)


# def varize(sympy_func) -> Callable[..., VarExpr]:
#     @functools.wraps(sympy_func)
#     def wrapper(*args, **kwargs):
#         type_: VarType | None = None
#         dependencies: list[Var] = []

#         sympy_args = []
#         for arg in args:
#             if isinstance(arg, VarExpr):
#                 sympy_args.append(arg.sympy_expr)
#                 dependencies.extend(arg.dependencies)
#                 if type_ is None:
#                     type_ = arg.type
#             elif isinstance(arg, Var):
#                 sympy_args.append(sympy.Symbol(arg.label.latex))
#                 dependencies.append(arg)
#                 if type_ is None:
#                     type_ = arg.type
#             else:
#                 sympy_args.append(arg)

#         sympy_kwargs = {}
#         for key, arg in kwargs.items():
#             if isinstance(arg, VarExpr):
#                 sympy_kwargs[key] = arg.sympy_expr
#                 dependencies.extend(arg.dependencies)
#                 if type_ is None:
#                     type_ = arg.type
#             elif isinstance(arg, Var):
#                 sympy_kwargs[key] = sympy.Symbol(arg.label.latex)
#                 dependencies.append(arg)
#                 if type_ is None:
#                     type_ = arg.type
#             else:
#                 sympy_kwargs[key] = arg

#         return VarExpr(
#             sympy_func(*sympy_args, **sympy_kwargs), type_ or object, dependencies
#         )

#     return wrapper


# sin = varize(sympy.sin)
# cos = varize(sympy.cos)
# tan = varize(sympy.tan)
# cot = varize(sympy.cot)

# asin = varize(sympy.asin)
# acos = varize(sympy.acos)
# atan = varize(sympy.atan)
# acot = varize(sympy.acot)

# sinh = varize(sympy.sinh)
# cosh = varize(sympy.cosh)
# tanh = varize(sympy.tanh)
# coth = varize(sympy.coth)

# exp = varize(sympy.exp)
# log = varize(sympy.log)

# sqrt = varize(sympy.sqrt)

# diff = varize(sympy.diff)
# integrate = varize(sympy.integrate)


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


class Var(Symbol["Var[T]", T], Generic[T]):
    def __init__(
        self,
        label: Any,
        type_: VarTypeLike[T] = object,
        metadata: MetadataLike = None,
        name: str | None = None,
    ) -> None:
        self._label = interpret_var_label(label)
        self.metadata = Metadata.interpret(metadata)
        self._type = VarType.interpret(type_)

        if name is None:
            name = self.label.ascii

        super().__init__(name, self._type.value_type)

    @property
    def label(self) -> Text:
        return self._label

    @property
    def type(self) -> VarType[T]:
        return self._type

    # Override evaluate method with parsing
    # def evaluate(self, *args, check_type: bool = True) -> T:
    #     result = super().evaluate(
    #         *args, check_type=False
    #     )  # Type checking is done by parse function
    #     return self.parse(result, check=check_type)

    def default(self) -> T | Undefined:
        return self._type.default()

    def has_default(self):
        return self.default() is not undefined

    @property
    def text(self) -> Text:
        return self.label

    @overload
    def format(self, value: Any, *, check: bool = True) -> Text:
        ...

    @overload
    def format(self, value: T, *, parse: Literal[False], check: bool = True) -> Text:
        ...

    def format(self, value: Any, *, parse: bool = True, check: bool = True) -> Text:
        if parse:
            value = self._type.parse(value)

        if check:
            self._type.check(value)

        return self._type.format(value)

    def parse(self, raw: Any, *, check: bool = True) -> T:
        result = self._type.parse(raw)
        if check:
            self._type.check(result)
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
            value = self._type.parse(value)

        if check:
            self._type.check(value)

        return self._type.get_plot_value(value)

    @overload
    def equal(self, a: Any, b: Any, *, check: bool = True) -> bool:
        ...

    @overload
    def equal(self, a: T, b: T, *, parse: Literal[False], check: bool = True) -> bool:
        ...

    def equal(self, a: Any, b: Any, *, parse: bool = True, check: bool = True) -> bool:
        if parse:
            a = self._type.parse(a)
            b = self._type.parse(b)

        if check:
            self._type.check(a)
            self._type.check(b)

        return self._type.equal(a, b)

    @overload
    def check(self, value: Any, *, parse: bool = True) -> None:
        ...

    @overload
    def check(self, value: T, *, parse: Literal[False]) -> None:
        ...

    def check(self, value: Any, *, parse: bool = True) -> None:
        if parse:
            value = self._type.parse(value)

        self._type.check(value)

    def entry(self, value: Any):
        from ..data import RecordEntry

        return RecordEntry(self, self.parse(value))

    def matches(self, query: VarQuery) -> bool:

        if is_wildcard(query) or query is self:
            return True

        if isinstance(query, str):
            if self._label.matches(query):
                return True

            if ":" in query:
                key, value = query.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key == "label":
                    return self._label.matches(value)
                if key == "type":
                    return self._type.matches(value)
                if key == "metadata":
                    return self.metadata.matches(value)

            return False

        if isinstance(query, dict):
            if "label" in query and not self._label.matches(query["label"]):
                return False
            if "type" in query and not self._type.matches(query["type"]):
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
            return self._label.matches(query.label)

        return False
        # raise TypeError(f"Cannot match var {self} by query of type {type(query).__name__}.")

    def eq(self, other: Any) -> RecordEntryCondition:
        return RecordEntryCondition.for_other(self, other, lambda x, y: x == y)

    def ne(self, other: Any) -> RecordEntryCondition:
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

    def __eq__(self, other: Any):
        return isinstance(other, Var) and self.matches(other)

    def __hash__(self):
        return id(self)

    def sub(self, *subscripts: Any):
        return Var(self.label.subscript(Text(", ").join(subscripts)), self.type)

    @overload
    def wrap(self, function_name: Any) -> Var[T]:
        ...

    @overload
    def wrap(self, function_name: Any, type_: VarTypeLike[S]) -> Var[S]:
        ...

    def wrap(self, function_name: Any, type_: VarTypeLike | None = None) -> Var:
        return Var(function_name + lang.par(self.label), type_ or self.type)

    def copy(
        self,
        *,
        label: Text | None = None,
        type: VarTypeLike[T] | None = None,
        metadata: Metadata | None = None,
    ):
        return Var(label or self.label, type or self.type, metadata or self.metadata)


def vars(labels: Iterable[Any] | str, type_: VarTypeLike[T] = object):
    if isinstance(labels, str):
        labels = labels.strip().split(" ")

    return [Var(label, type_) for label in labels]


VarQuery = Union[Var[T], Wildcard, Any]


def is_var_query(query: Any) -> TypeGuard[VarQuery]:
    return True


class RecordEntryCondition(Generic[T]):
    @staticmethod
    def for_other(
        var: Var, other: Any, check: Callable[[Any, Any], bool]
    ) -> RecordEntryCondition:
        if isinstance(other, Var):
            return RecordEntryCondition([var, other], lambda x, y: check(x, y))
        else:
            return RecordEntryCondition([var], lambda x: check(x, other))

    def __init__(self, vars: Iterable[Var[T]], check: Callable[..., bool]) -> None:
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
    def get_value(self, record: "m_record.Record") -> T:
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

    def get_value(self, record: "m_record.Record") -> T:
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
        expr: ExprLike[Var, T],
    ) -> None:
        self._expr = Expr.interpret(expr)
        # self._expr = VarExpr.interpret(expr)
        super().__init__(label, self._expr.value_type)

    def _compute_for_args(self, *args) -> T:
        result = self._expr.evaluate(
            *args, check_type=False
        )  # Type checking is done by parse function
        return self.parse(result)

    def get_value(self, record: "m_record.Record") -> T:
        # print(
        #     "get value of", self.equality_text.ascii, "with symbols", self._expr.symbols
        # )
        args = [record[var] for var in self._expr.symbols]
        # print("-> args:", args)
        return self._compute_for_args(*args)

    def get_dependencies(self) -> list[Var]:
        return self._expr.symbols

    @overload
    def __call__(self, record: "m_record.Record", /) -> T:
        ...

    @overload
    def __call__(self, *args) -> T:
        ...

    def __call__(self, *args, as_args: bool = False) -> T:
        if not as_args and len(args) == 1 and m_record.is_record_like(args[0]):
            return self.get_value(m_record.Record.interpret(args[0]))
        else:
            return self._compute_for_args(*args)

    @property
    def functional_label(self):
        return self.label + lang.par(
            Text(", ").join(var.name for var in self._expr.symbols)
        )

    @property
    def equality_text(self):
        return self.functional_label + " = " + self._expr.text


class FunctionComputed(DerivedVar[T]):
    def __init__(
        self,
        label: Any,
        function: Callable[["m_record.Record"], T],
        type_: VarTypeLike[T] | None = None,
        dependencies: Iterable[Var] | None = None,
    ) -> None:
        self._function = function

        if dependencies is None:
            observer_record = m_record.ObserverRecord()
            self._function(observer_record)
            dependencies = observer_record.accessed_vars

        self._dependencies = list(dependencies)

        super().__init__(label, type_ or infer_return_type(function))

    def get_value(self, record: "m_record.Record") -> T:
        return self._function(record)

    def get_dependencies(self) -> list[Var]:
        return self._dependencies

    def __call__(self, record: "m_record.RecordLike", /) -> T:
        return self.get_value(m_record.Record.interpret(record))

    @property
    def functional_label(self):
        return self.label + lang.par(
            Text(", ").join(var.name for var in self._dependencies)
        )
