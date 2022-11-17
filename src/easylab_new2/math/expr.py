from __future__ import annotations
from abc import ABC, abstractmethod
import math
from types import MappingProxyType
from typing import Any, Generic, Iterable, TypeVar, Union
from ..dispatch import *
from typing_extensions import Self

T = TypeVar("T")
S = TypeVar("S")

EvaluationContextLike = Union[
    "EvaluationContext",
    dict[Union[str, "Symbol"], Any],
    dict[str, Any],
    dict["Symbol", Any],
    None,
]


class EvaluationContext:
    @staticmethod
    def interpret(input: EvaluationContextLike) -> EvaluationContext:
        if input is None:
            return EvaluationContext({})
        elif isinstance(input, EvaluationContext):
            return input
        elif isinstance(input, dict):
            return EvaluationContext(
                {Symbol.interpret(symbol): value for symbol, value in input.items()}
            )
        else:
            raise TypeError(f"Cannot interpret {input!r} as EvaluationContext")

    def __init__(
        self,
        locals: dict[Symbol, Any],
    ) -> None:
        assert isinstance(locals, dict)
        self._locals = locals.copy()

    def get_locals(self) -> MappingProxyType[Symbol, Any]:
        return MappingProxyType(self._locals)

    def get_values(self) -> MappingProxyType[Symbol, Any]:
        return MappingProxyType(self._locals)

    def __getitem__(self, symbol: Symbol | str):
        symbol = Symbol.interpret(symbol)

        if symbol in self._locals:
            return self._locals[symbol]
        else:
            raise KeyError(
                f"Symbol {symbol!r} not found in context. Available symbols are: "
                + ", ".join(str(symbol) for symbol in self._locals)
            )

    # def __setitem__(self, symbol: Symbol[S], value: S) -> None:
    #     self.locals[symbol] = value

    def __contains__(self, symbol: Symbol | str) -> bool:
        symbol = Symbol.interpret(symbol)

        return symbol in self._locals

    # def update(self, locals: dict[Symbol, Any]) -> None:
    #     self.locals.update(locals)

    def copy(self) -> Self:
        return EvaluationContext(self._locals.copy())  # type: ignore

    def extend(self, other: EvaluationContextLike) -> Self:
        other = EvaluationContext.interpret(other)
        result = self.copy()
        result._locals.update(other._locals)
        return result

    def create_child(self, expr: Expr[S]) -> ChildEvaluationContext[S]:
        return ChildEvaluationContext(self, expr, {})


class ChildEvaluationContext(EvaluationContext, Generic[T]):
    def __init__(
        self,
        parent: EvaluationContext,
        expr: Expr[T],
        locals: dict[Symbol, Any],
    ) -> None:
        assert isinstance(parent, EvaluationContext)
        assert isinstance(expr, Expr)
        self.parent = parent
        self.expr = expr
        super().__init__(locals)

    def get_values(self) -> MappingProxyType[Symbol, Any]:
        return MappingProxyType(self.parent.get_values() | self._locals)

    def __getitem__(self, symbol: Symbol | str):
        symbol = Symbol.interpret(symbol)
        if symbol in self._locals:
            return self._locals[symbol]
        else:
            return self.parent[symbol]

    def __contains__(self, symbol: Symbol | str) -> bool:
        symbol = Symbol.interpret(symbol)
        return symbol in self._locals or symbol in self.parent

    def copy(self) -> Self:
        return ChildEvaluationContext(self.parent, self.expr, self._locals.copy())  # type: ignore


# class ExprType(Generic[T]):
#     def __init__(self, value_type: type[T], zero: T | None = None, one: T | None = None, neg: Callable[[T], T] | None = None):
#         self.value_type = value_type
#         self.zero = zero
#         self.one = one
#         self.neg = neg

# _expression_types = {
#     int: ExprType(int, 0, 1, lambda x: -x),
#     float: ExprType(float, 0.0, 1.0, lambda x: -x),
#     complex: ExprType(complex, complex(0), complex(1), lambda x: -x),
#     bool: ExprType(bool, False, True, lambda x: not x),
#     str: ExprType(str),
# }


ExprLike = Union["Expr[T]", T]


class Expr(ABC, Generic[T]):
    @staticmethod
    def interpret(input: ExprLike[T]) -> Expr[T]:
        if isinstance(input, Expr):
            return input
        else:
            return ValueExpr(input)

    def __init__(self, dependencies: Iterable[Expr]) -> None:
        self.dependencies: list[Symbol] = []
        for dependency in dependencies:
            if isinstance(dependency, Symbol):
                self.dependencies.append(dependency)
            else:
                self.dependencies.extend(dependency.dependencies)

    @abstractmethod
    def evaluate_in_own_context(self, context: ChildEvaluationContext[T]) -> T:
        ...

    def evaluate(
        self,
        context: EvaluationContextLike = None,
    ) -> T:
        context = EvaluationContext.interpret(context)
        own_context = context.create_child(self)
        return self.evaluate_in_own_context(own_context)

    def diff(self, symbol: Symbol) -> Expr[T]:
        raise NotImplementedError(f"Cannot differentiate {type(self).__name__}.")

    def anti_diff(self, symbol: Symbol) -> Expr[T]:
        raise NotImplementedError(f"Cannot anti-differentiate {type(self).__name__}.")

    def integrate(
        self,
        symbol: Symbol,
        lower: ExprLike[T],
        upper: ExprLike[T],
    ) -> Expr[T]:
        primitive = self.anti_diff(symbol)

        return WithLocals({symbol: upper}, primitive) - WithLocals(
            {symbol: lower}, primitive
        )

    def __call__(
        self, args: dict[str | Symbol, ExprLike] | None = None, /, **kwargs: ExprLike
    ) -> Expr[T]:
        return FunctionCallExpr(self, (args or {}) | kwargs)

    def __add__(self, other: ExprLike[T]) -> Expr[T]:
        return AddExpr(self, other)

    def __radd__(self, other: ExprLike[T]) -> Expr[T]:
        return AddExpr(other, self)

    def __mul__(self, other: ExprLike[T]) -> Expr[T]:
        return MulExpr(self, other)

    def __rmul__(self, other: ExprLike[T]) -> Expr[T]:
        return MulExpr(other, self)

    def __neg__(self) -> Expr[T]:
        return NegExpr(self)

    def __sub__(self, other: ExprLike[T]) -> Expr[T]:
        return SubExpr(self, other)


class Symbol(Expr[T]):
    @staticmethod
    def interpret(input: Symbol[T] | str) -> Symbol[T]:
        if isinstance(input, Symbol):
            return input
        else:
            return Symbol(input)

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__([])

    def evaluate_in_own_context(self, context: ChildEvaluationContext[T]) -> T:
        if self.name in context:
            return context[self.name]
        else:
            raise ValueError(f"Symbol {self.name!r} is not defined in context.")

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"

    def __str__(self) -> str:
        return self.name


class WithLocals(Expr[T]):
    def __init__(
        self, locals: dict[str | Symbol, ExprLike], inner: ExprLike[T]
    ) -> None:
        self.locals = {
            Symbol.interpret(symbol): Expr.interpret(value)
            for symbol, value in locals.items()
        }
        self.inner = Expr.interpret(inner)
        super().__init__([self.inner])

    def evaluate_in_own_context(self, context: ChildEvaluationContext[T]) -> T:
        return self.inner.evaluate(context.extend(self.locals))


class ValueExpr(Expr[T]):
    def __init__(self, value: T) -> None:
        self.value = value
        super().__init__([])

    def evaluate_in_own_context(self, context: ChildEvaluationContext[T]) -> T:
        return self.value


ans_symbol = Symbol("ans")


class FunctionCallExpr(Expr[T]):
    def __init__(self, expr1: ExprLike[T], args: dict[Symbol | str, ExprLike]) -> None:
        self.function = Expr.interpret(expr1)
        self.args = {
            Symbol.interpret(symbol): Expr.interpret(value)
            for symbol, value in args.items()
        }
        super().__init__([self.function, *self.args.values()])

    def evaluate_in_own_context(self, context: ChildEvaluationContext[T]) -> T:
        return self.function.evaluate(context.extend(self.args))

    def diff(self, symbol: Symbol) -> Expr[T]:
        return AddExpr(
            *(
                self.function.diff(arg_symbol) * arg.diff(symbol)
                for arg_symbol, arg in self.args.items()
            )
        )


class AddExpr(Expr[T]):
    def __init__(self, *summands: ExprLike[T]) -> None:
        if len(summands) == 0:
            raise ValueError("AddExpr must have at least one summand.")

        self.summands = [Expr.interpret(summand) for summand in summands]
        super().__init__(self.summands)

    def evaluate_in_own_context(self, context: ChildEvaluationContext[T]) -> T:
        return add(*(summand.evaluate(context) for summand in self.summands))


class MulExpr(Expr[T]):
    def __init__(self, *factors: ExprLike[T]) -> None:
        if len(factors) == 0:
            raise ValueError("MulExpr must have at least one factor.")

        self.factors = [Expr.interpret(factor) for factor in factors]
        super().__init__(self.factors)

    def evaluate_in_own_context(self, context: ChildEvaluationContext[T]) -> T:
        return mul(*(factor.evaluate(context) for factor in self.factors))


class NegExpr(Expr[T]):
    def __init__(self, expr: ExprLike[T]) -> None:
        self.expr = Expr.interpret(expr)
        super().__init__([self.expr])

    def evaluate_in_own_context(self, context: ChildEvaluationContext[T]) -> T:
        return neg(self.expr.evaluate(context))


main_evaluation_context = EvaluationContext({})

pi = Symbol("pi")
e = Symbol("e")
numeric_evaluation_context = EvaluationContext({pi: math.pi, e: math.e})

x = Symbol("x")
y = Symbol("y")
expr = WithLocals(
    {
        x: AddExpr(1, 2, 3),
    },
    x + y,
)
expr.evaluate()
