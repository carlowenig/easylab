from copy import copy
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Union,
    cast,
)
from typing_extensions import Self
import numpy as np
import sympy

from ..lang.text import Text, TextInput


def create_expr_function(expr: sympy.Expr, dependencies: Iterable["ExprObject"] = []):
    return sympy.lambdify([dep._expr for dep in dependencies], expr)


def evaluate_expr(expr: sympy.Expr, dependency_values: dict["ExprObject", Any] = {}):
    f = create_expr_function(expr, dependency_values.keys())
    return f(*dependency_values.values())


_numpy_to_sympy_name_mappings = {
    "add": "Add",
    "subtract": "Sub",
    "multiply": "Mul",
    "divide": "Div",
}


class ExprObject:
    _expr: sympy.Expr
    _dependencies: tuple["ExprObject", ...]

    def __init__(
        self, expr: Union[str, sympy.Expr], dependencies: Iterable["ExprObject"] = []
    ) -> None:
        self._expr = sympy.Symbol(expr) if isinstance(expr, str) else expr
        self._dependencies = tuple(dict.fromkeys(dependencies))  # Remove duplicates

    @classmethod
    def from_expr(cls, expr: sympy.Expr, dependencies: list["ExprObject"]) -> Self:
        expr = sympy.simplify(expr)

        # Do not use the subclasses __init__ method, because this will probably be used to create
        # the object without an expression. Use __init_from_expr__ instead.
        obj: Self = object.__new__(cls, expr, dependencies)  # type: ignore
        ExprObject.__init__(obj, expr, dependencies)

        obj.__init_from_expr__()
        return obj

    @property
    def expr(self) -> sympy.Expr:
        return self._expr

    @property
    def text(self) -> Text:
        return Text(
            str(self._expr),
            unicode=sympy.pretty(self._expr, use_unicode=True, wrap_line=False),
            latex=sympy.latex(self._expr),
        )

    @property
    def dependencies(self) -> tuple["ExprObject", ...]:
        return self._dependencies

    def __init_from_expr__(self):
        pass

    def create_eval_function(self) -> Callable:
        return sympy.lambdify([dep._expr for dep in self._dependencies], self._expr)

    def eval(self, *dependency_values):
        return self.create_eval_function()(*dependency_values)

    def __check_operation__(self, name: str, *args) -> None:
        pass

    def __run_operation__(self, name: str, *args) -> Self:
        self.__check_operation__(name, *args)

        if len(self._dependencies) == 0:
            dependencies = [cast(ExprObject, self)]
        else:
            dependencies = list(self._dependencies)

        sympy_args = []
        for arg in args:
            if isinstance(arg, ExprObject):
                sympy_args.append(arg._expr)

                if len(arg._dependencies) == 0:
                    dependencies.append(arg)
                for dep in arg._dependencies:
                    if dep not in dependencies:
                        dependencies.append(dep)
            else:
                sympy_args.append(arg)

        return type(self).from_expr(
            getattr(self._expr, name)(*sympy_args), dependencies
        )

    def __add__(self, other: Any) -> Self:
        return self.__run_operation__("__add__", other)

    def __radd__(self, other: Any) -> Self:
        return self.__run_operation__("__radd__", other)

    def __sub__(self, other: Any) -> Self:
        return self.__run_operation__("__sub__", other)

    def __rsub__(self, other: Any) -> Self:
        return self.__run_operation__("__rsub__", other)

    def __mul__(self, other: Any) -> Self:
        return self.__run_operation__("__mul__", other)

    def __rmul__(self, other: Any) -> Self:
        return self.__run_operation__("__rmul__", other)

    def __truediv__(self, other: Any) -> Self:
        return self.__run_operation__("__truediv__", other)

    def __rtruediv__(self, other: Any) -> Self:
        return self.__run_operation__("__rtruediv__", other)

    def __pow__(self, exp: Any) -> Self:
        return self.__run_operation__("__pow__", exp)

    def __rpow__(self, other: Any) -> Self:
        return self.__run_operation__("__rpow__", other)

    def __neg__(self) -> Self:
        return self.__run_operation__("__neg__")

    def __pos__(self) -> Self:
        return self.__run_operation__("__pos__")

    def __abs__(self) -> Self:
        return self.__run_operation__("__abs__")

    # Specify __array__ method to allow ufuncs to work with VarLike objects
    def __array__(self, dtype=None):
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        if len(self._dependencies) == 0:
            dependencies = [cast(ExprObject, self)]
        else:
            dependencies = list(self._dependencies)
        sympy_args = []

        for input in inputs:
            if isinstance(input, ExprObject):
                sympy_args.append(input.expr)

                if len(input._dependencies) == 0:
                    dependencies.append(input)
                for dep in input._dependencies:
                    if dep not in dependencies:
                        dependencies.append(dep)
            else:
                # Just use the input itself (e.g. for numbers)
                sympy_args.append(input)

        name = _numpy_to_sympy_name_mappings.get(ufunc.__name__, ufunc.__name__)
        expr = getattr(sympy, name)(*sympy_args)
        return type(self).from_expr(expr, dependencies)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, ExprObject)
            and type(self) == type(other)
            and sympy.simplify(self._expr - other._expr) == 0
            and self._dependencies == other._dependencies
        )

    def __hash__(self) -> int:
        return hash(self._expr)

    def __str__(self) -> str:
        return str(self._expr)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._expr}, dependencies={self._dependencies})"


class LabeledExprObject(ExprObject):
    _label: Text

    def __init__(self, label: TextInput) -> None:
        self.label = label
        super().__init__(sympy.Symbol(self.label.default), [])

    def __init_from_expr__(self):
        self._label = super().text

    @property
    def label(self) -> Text:
        return self._label

    @label.setter
    def label(self, label: TextInput) -> None:
        self._label = Text.parse(label)

    def __or__(self, label: TextInput):
        result = copy(self)
        result.label = label
        return result

    @property
    def text(self) -> Text:
        return self._label

    def __str__(self) -> str:
        return self._label.default
