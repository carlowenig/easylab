from __future__ import annotations
import functools
import re
from typing import Any, Callable, Generic, Iterable, TypeVar, Union, cast
from typing_extensions import Self
import sympy
from ..lang import Text


V = TypeVar("V")


S = TypeVar("S", bound="Symbol")
ExprLike = Union["Expr[S, V]", S, V, Any]


class Expr(Generic[S, V]):
    @staticmethod
    def interpret(
        input: ExprLike[S, V], symbol_type_hint: type[S] | None = None
    ) -> Expr[S, V]:
        if isinstance(input, Expr):
            return input
        elif isinstance(input, sympy.Expr):
            return Expr(input)  # type: ignore
        elif isinstance(input, int):
            return ValueExpr(input, sympy.Integer(input), symbol_type_hint, int)  # type: ignore
        elif isinstance(input, float):
            return ValueExpr(input, sympy.Float(input), symbol_type_hint, float)  # type: ignore
        elif isinstance(input, str):
            return Expr(sympy.sympify(input), symbol_type=symbol_type_hint)  # type: ignore
        else:
            raise TypeError(f"Cannot interpret {type(input)} as Expr")

    @staticmethod
    def parse(
        s: str,
        symbols: Iterable[S] = [],
        symbol_type: type[S] | None = None,
        value_type: type[V] | None = None,
    ) -> Expr:
        # Replace symbol labels by their names to allow sympy parsing
        # TODO: Using replace here is probably error prone. Try to find a better way.
        for symbol in symbols:
            for target_label in symbol.label._target_strings.values():
                s = s.replace(target_label, symbol.name)

        from sympy.parsing.sympy_parser import parse_expr

        sympy_expr = parse_expr(s)

        return Expr(sympy_expr, symbols, symbol_type, value_type)

    def __init__(
        self,
        sympy_expr: sympy.Expr,
        symbols: Iterable[S] = [],
        symbol_type: type[S] | None = None,
        value_type: type[V] | None = None,
    ) -> None:
        if not isinstance(sympy_expr, sympy.Expr):
            raise TypeError(f"sympy_expr must be sympy.Expr, not {type(sympy_expr)}")

        self._sympy_expr = sympy_expr
        self._symbols = list(symbols)

        if symbol_type is None:
            if len(self._symbols) == 0:
                symbol_type = cast(type[S], object)
            else:
                symbol_type = type(self._symbols[0])

        if value_type is None:
            if len(self._symbols) == 0:
                value_type = cast(type[V], object)
            else:
                value_type = cast(type[V], self._symbols[0].value_type)

        self._symbol_type = cast(type[S], symbol_type)
        self._value_type = cast(type[V], value_type)

    @property
    def sympy_expr(self):
        return self._sympy_expr

    @property
    def symbol_type(self) -> type[S]:
        return self._symbol_type  # type: ignore # TODO: Why is this needed?

    @property
    def value_type(self) -> type[V]:
        return self._value_type

    @property
    def symbols(self) -> list[S]:
        return self._symbols  # type: ignore # TODO: Why is this needed?

    @functools.cache
    def _create_evaluator(self, **lambdify_kwargs) -> Callable:
        args = []
        for symbol in self.symbols:
            if symbol.sympy_expr not in args:
                args.append(symbol.sympy_expr)
            else:
                args.append(sympy.Dummy())
        return sympy.lambdify(
            args,
            self.sympy_expr,
            **lambdify_kwargs,
        )

    def evaluate(self, *args, check_type: bool = True) -> V:
        evaluator = self._create_evaluator()
        result = evaluator(*args)

        if isinstance(result, sympy.Expr) and not issubclass(
            self.value_type, sympy.Expr
        ):
            args_dict = {symbol.name: arg for symbol, arg in zip(self.symbols, args)}
            raise ValueError(
                f"Expr {self} could not be fully evaluated for args {args_dict}. Result was {result}."
            )

        if check_type and not isinstance(result, self._value_type):
            raise ValueError(
                f"Expr evaluated to invalid type. Expected {self._value_type}, got result {result!r} which is of type {type(result)}."
            )

        return cast(V, result)

    def transform(self, f: Callable[[sympy.Expr], Any]):
        return Expr(
            f(self._sympy_expr), self._symbols, self._symbol_type, self._value_type
        )

    def combine(self, other: ExprLike, f: Callable[[sympy.Expr, sympy.Expr], Any]):
        other = Expr.interpret(other)
        return Expr(
            f(self._sympy_expr, other._sympy_expr),
            self._symbols + other._symbols,
            self._symbol_type,
            self._value_type,
        )

    @property
    def text(self):
        latex = sympy.latex(self._sympy_expr, mode="inline")
        return Text(str(self._sympy_expr), latex=latex)

    def __repr__(self):
        return self.text.ascii

    def __str__(self):
        return self.text.ascii

    def __add__(self, other: ExprLike):
        return self.combine(other, lambda a, b: a + b)

    def __radd__(self, other: ExprLike):
        return Expr.interpret(other) + self

    def __sub__(self, other: ExprLike):
        return self.combine(other, lambda a, b: a - b)

    def __rsub__(self, other: ExprLike):
        return Expr.interpret(other) - self

    def __mul__(self, other: ExprLike):
        return self.combine(other, lambda a, b: a * b)

    def __rmul__(self, other: ExprLike):
        return Expr.interpret(other) * self

    def __truediv__(self, other: ExprLike):
        return self.combine(other, lambda a, b: a / b)

    def __rtruediv__(self, other: ExprLike):
        return Expr.interpret(other) / self

    def __pow__(self, other: ExprLike):
        return self.combine(other, lambda a, b: a**b)

    def __rpow__(self, other: ExprLike):
        return Expr.interpret(other) ** self

    def __neg__(self):
        return self.transform(lambda a: -a)

    def __pos__(self):
        return self

    def __abs__(self):
        return self.transform(abs)


# def sanitize_symbol_name(name: str) -> str:
#     name = re.sub(r"\W|^(?=\d)", "", name)
#     while name in sympy.__dict__:
#         name = "S_{" + name + "}"
#     return name

_symbol_sympy_expr_count = 0


def create_sympy_lab_symbol():
    global _symbol_sympy_expr_count
    _symbol_sympy_expr_count += 1
    return sympy.Symbol(f"labsymbol_{_symbol_sympy_expr_count}")


class Symbol(Expr[S, V]):
    """
    Type parameter S should be the type of the symbol itself.
    """

    sympy_expr: sympy.Symbol

    def __init__(
        self,
        label: Any,
        value_type: type[V],
        *,
        sympy_expr: sympy.Symbol | None = None,
    ):
        self._label = Text.interpret(label)
        self._value_type = value_type

        super().__init__(
            sympy_expr if sympy_expr is not None else create_sympy_lab_symbol(),
            symbols=[self],  # type: ignore
            symbol_type=type(self),  # type: ignore
            value_type=value_type,
        )

    @property
    def name(self) -> str:
        return self.sympy_expr.name

    @property
    def label(self) -> Text:
        return self._label

    @property
    def text(self):
        return self._label

    def matches(self, query: Any):
        return self._label.matches(query) or (
            isinstance(query, str) and query == self.name
        )


class ValueExpr(Expr[S, V]):
    def __init__(
        self,
        value: V,
        sympy_expr: sympy.Expr | None = None,
        symbol_type: type[S] | None = None,
        value_type: type[V] | None = None,
    ):
        self._value = value

        super().__init__(
            sympy_expr if sympy_expr is not None else sympy.sympify(value),
            symbols=[],
            symbol_type=symbol_type,
            value_type=value_type or type(value),
        )


# class Symbol(Expr[B, V]):
#     def __init__(
#         self, label: Any, base: B, *, fallback: V | Undefined = undefined, base_type: type[B] | None = None, value_type: type[V] = object
#     ) -> None:
#         self._label = Text.interpret(label)
#         self._base = base
#         self._fallback = fallback
#         super().__init__(sympy.Symbol(self._label.latex), type_, [self])

#     @property
#     def label(self) -> Text:
#         return self._label

#     @property
#     def fallback(self):
#         return self._fallback

#     @property
#     def text(self):
#         return self.label

#     def evaluate(self, value) -> T:
#         return value


# class Constant(Expr[T]):
#     def __init__(self, value: T, type_: type[T] | None = None) -> None:
#         self._value = value
#         super().__init__(value, type_ or type(value))

#     @property
#     def value(self) -> T:
#         return self._value

#     @property
#     def text(self):
#         return text(self.value)

#     def evaluate(self) -> T:
#         return self.value


def expr_func_from_sympy(sympy_func) -> Callable[..., Any]:
    @functools.wraps(sympy_func)
    def wrapper(*args, **kwargs):
        symbol_type: type | None = None
        dependencies: list[Any] = []

        sympy_args = []
        for arg in args:
            if isinstance(arg, Expr):
                sympy_args.append(arg.sympy_expr)
                dependencies.extend(arg.symbols)
                if symbol_type is None:
                    symbol_type = arg.symbol_type
            else:
                sympy_args.append(arg)

        sympy_kwargs = {}
        for key, arg in kwargs.items():
            if isinstance(arg, Expr):
                sympy_kwargs[key] = arg.sympy_expr
                dependencies.extend(arg.symbols)
                if symbol_type is None:
                    symbol_type = arg.symbol_type
            else:
                sympy_kwargs[key] = arg

        return Expr(
            sympy_func(*sympy_args, **sympy_kwargs),
            dependencies,
            symbol_type,
        )

    return wrapper


sin = expr_func_from_sympy(sympy.sin)
cos = expr_func_from_sympy(sympy.cos)
tan = expr_func_from_sympy(sympy.tan)
cot = expr_func_from_sympy(sympy.cot)

asin = expr_func_from_sympy(sympy.asin)
acos = expr_func_from_sympy(sympy.acos)
atan = expr_func_from_sympy(sympy.atan)
acot = expr_func_from_sympy(sympy.acot)

sinh = expr_func_from_sympy(sympy.sinh)
cosh = expr_func_from_sympy(sympy.cosh)
tanh = expr_func_from_sympy(sympy.tanh)
coth = expr_func_from_sympy(sympy.coth)

exp = expr_func_from_sympy(sympy.exp)
log = expr_func_from_sympy(sympy.log)

sqrt = expr_func_from_sympy(sympy.sqrt)

diff = expr_func_from_sympy(sympy.diff)
integrate = expr_func_from_sympy(sympy.integrate)
