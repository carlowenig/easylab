from __future__ import annotations
from abc import ABC, abstractmethod
import math
from types import MappingProxyType
from typing import Any, Generic, Iterable, TypeVar, Self, Union
from ..dispatch import *

T = TypeVar("T")
S = TypeVar("S")

EvaluationContextLike = Union["EvaluationContext", dict[str | "Symbol", Any]]


def symbol_name(symbol: str | Symbol) -> str:
    if isinstance(symbol, Symbol):
        return symbol.name
    else:
        return symbol


class EvaluationContext:
    @staticmethod
    def interpret(input: EvaluationContextLike) -> EvaluationContext:
        if isinstance(input, EvaluationContext):
            return input
        else:
            return EvaluationContext(input)

    def __init__(
        self,
        locals: dict[str | Symbol, Any],
    ) -> None:
        assert isinstance(locals, dict)
        self._locals = {symbol_name(symbol): value for symbol, value in locals.items()}

    def get_locals(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(self._locals)

    def get_values(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(self._locals)

    def __getitem__(self, symbol: Symbol[S] | str) -> S:
        symbol = symbol_name(symbol)

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
        return symbol_name(symbol) in self._locals

    # def update(self, locals: dict[Symbol, Any]) -> None:
    #     self.locals.update(locals)

    def copy(self) -> Self:
        return EvaluationContext(self._locals.copy())  # type: ignore

    def extend(self, locals: dict[str, Any]) -> Self:
        result = self.copy()
        result._locals.update(locals)
        return result

    def create_child(self, expr: Expr[S]) -> ChildEvaluationContext[S]:
        return ChildEvaluationContext(self, expr, {})


class ChildEvaluationContext(EvaluationContext, Generic[T]):
    def __init__(
        self,
        parent: EvaluationContext,
        expr: Expr[T],
        locals: dict[str | Symbol, Any],
    ) -> None:
        assert isinstance(parent, EvaluationContext)
        assert isinstance(expr, Expr)
        self.parent = parent
        self.expr = expr
        super().__init__(locals)

    def get_values(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(self.parent.get_values() | self._locals)

    def __getitem__(self, symbol: Symbol[S] | str) -> S:
        symbol = symbol_name(symbol)

        if symbol in self._locals:
            return self._locals[symbol]
        else:
            return self.parent[symbol]

    def __contains__(self, symbol: Symbol | str) -> bool:
        symbol = symbol_name(symbol)
        return symbol in self._locals or symbol in self.parent

    def copy(self) -> Self:
        return ChildEvaluationContext(self.parent, self.expr, self._locals.copy())  # type: ignore


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
        parent_context: EvaluationContext | dict[Symbol, Any] = {},
    ) -> T:
        if isinstance(parent_context, dict):
            parent_context = main_evaluation_context.extend(parent_context)

        own_context = parent_context.create_child(self)
        return self.evaluate_in_own_context(own_context)

    def __rshift__(self, other: ExprLike[S]) -> Expr[S]:
        return ChainedExpr(self, other)

    def __rrshift__(self, other: ExprLike[S]) -> Expr[T]:
        return ChainedExpr(other, self)

    def __add__(self, other: ExprLike[T]) -> Expr[T]:
        return AddExpr(self, other)

    def __radd__(self, other: ExprLike[T]) -> Expr[T]:
        return AddExpr(other, self)

    def __mul__(self, other: ExprLike[T]) -> Expr[T]:
        return MulExpr(self, other)

    def __rmul__(self, other: ExprLike[T]) -> Expr[T]:
        return MulExpr(other, self)


class Symbol(Expr[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__([])

    def evaluate_in_own_context(self, context: ChildEvaluationContext[T]) -> T:
        if self in context:
            return context[self]
        else:
            raise ValueError(f"Symbol {self.name!r} is not defined in context.")

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"

    def __str__(self) -> str:
        return self.name


class WithLocals(Expr[T]):
    def __init__(self, locals: dict[Symbol, ExprLike], inner: ExprLike[T]) -> None:
        self.locals = {
            symbol: Expr.interpret(value) for symbol, value in locals.items()
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


class ChainedExpr(Expr[T], Generic[S, T]):
    def __init__(self, expr1: ExprLike[S], expr2: ExprLike[T]) -> None:
        self.expr1 = Expr.interpret(expr1)
        self.expr2 = Expr.interpret(expr2)
        super().__init__([self.expr1, self.expr2])

    def evaluate_in_own_context(self, context: ChildEvaluationContext[T]) -> T:
        ans = self.expr1.evaluate(context)
        return self.expr2.evaluate(context.extend({ans_symbol: ans}))


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
