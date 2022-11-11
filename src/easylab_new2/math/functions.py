from __future__ import annotations
from abc import ABC, abstractmethod
import inspect
from typing import Any, Callable, Generic, TypeVar, Union
from ..dispatch import *
from .methods import *

T = TypeVar("T")

FunctionLike = Union["Function[T]", Callable[..., T], Any]


class Function(ABC, Generic[T], DispatchedOperators):
    @staticmethod
    def interpret(input: FunctionLike[T]) -> Function[T]:
        if isinstance(input, Function):
            return input
        elif callable(input):
            return CustomFunction(input)
        else:
            return ValueFunction(input)

    def __init__(self, name: str, arg_names: tuple[str, ...]) -> None:
        self.name = name
        self.arg_names = arg_names

    @abstractmethod
    def evaluate(self, **arg_functions: Function) -> T:
        pass

    def __call__(self, *args, **kwargs) -> T:
        arg_functions = self.get_arg_functions(*args, **kwargs)
        return self.evaluate(**arg_functions)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, arg_names={self.arg_names!r})"
        )

    def __str__(self) -> str:
        return self.name + "(" + ", ".join(self.arg_names) + ")"

    def get_arg_functions(self, *args, **kwargs):
        arg_functions: dict[str, Function] = {}

        if len(args) > len(self.arg_names):
            raise TypeError(
                f"Too many positional arguments for function {self.name!r}."
            )

        if any(name not in self.arg_names for name in kwargs):
            raise TypeError(f"Unknown keyword arguments for function {self.name!r}.")

        for i, name in enumerate(self.arg_names):
            if i < len(args):
                arg_functions[name] = Function.interpret(args[i])
            elif name in kwargs:
                arg_functions[name] = Function.interpret(kwargs[name])
            else:
                raise TypeError(f"Missing argument {name} for function {self.name}.")

        return arg_functions

    def str_for_args(self, *args, **kwargs):
        named_args = self.get_arg_functions(*args, **kwargs)
        return (
            self.name
            + "("
            + ", ".join(f"{name}={arg}" for name, arg in named_args.items())
            + ")"
        )


class CustomFunction(Function[T]):
    def __init__(
        self,
        func: Callable[..., T],
        name: str | None = None,
        arg_names: tuple[str, ...] | None = None,
    ) -> None:
        self.func = func
        params = inspect.signature(func).parameters
        for param in params.values():
            if param.kind not in [
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ]:
                raise TypeError("func must only have keyword arguments.")

        super().__init__(
            name if name is not None else func.__name__,
            arg_names
            if arg_names is not None
            else tuple(inspect.signature(func).parameters.keys()),
        )

    def evaluate(self, **arg_functions: Function) -> T:
        return self.func(**arg_functions)


class SetArgs(Function[T]):
    def __init__(
        self, function: FunctionLike[T], **arg_functions: FunctionLike[Any]
    ) -> None:
        self.function = Function.interpret(function)
        self.arg_functions = {
            name: Function.interpret(arg) for name, arg in arg_functions.items()
        }

        arg_names = list(function.arg_names)
        for name in self.arg_functions:
            arg_names.remove(name)

        super().__init__(function.str_for_args(*self.arg_functions), tuple(arg_names))

    def evaluate(self, **arg_functions: Function) -> T:
        for name in arg_functions:
            if name in self.arg_functions:
                raise TypeError(
                    f"Argument {name} is already set for function {self.name!r}."
                )

        return self.function(**(arg_functions | self.arg_functions))


class ValueFunction(Function[T]):
    def __init__(self, value: T) -> None:
        self.value = value
        super().__init__(name=f"{value}", arg_names=())

    def evaluate(self) -> T:
        return self.value


class SumFunction(Function[T]):
    def __init__(self, *summands: FunctionLike[T]) -> None:
        if len(summands) == 0:
            raise ValueError("Sum must have at least one summand.")
        self.summands = tuple(Function.interpret(summand) for summand in summands)
        super().__init__("sum", ())

    def evaluate(self) -> T:
        result = self.summands[0]()
        for summand in self.summands[1:]:
            result = add(result, summand())
        return result


class DiffFunction(Function[T]):
    def __init__(self, function: FunctionLike[T]):
        self.function = Function.interpret(function)

        if len(self.function.arg_names) != 1:
            raise ValueError("Differentiated function must have exactly one argument.")

        super().__init__(f"diff({function.name})", self.function.arg_names)

    def evaluate(self, arg: Any) -> T:
        return diff(self.function, arg)
