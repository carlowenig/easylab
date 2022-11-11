from abc import abstractclassmethod, abstractmethod
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
from sympy.printing.str import StrPrinter

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


class ExprPrinter(StrPrinter):
    def _print_Pow(self, expr, rational=False):
        return super()._print_Pow(expr, rational).replace("**", "^")


class ExprObject:
    _label: Optional[Text]
    _expr: Optional[sympy.Expr]
    _dependencies: tuple["ExprObject", ...]

    def __init__(
        self,
        *,
        expr: Optional[sympy.Expr] = None,
        label: Optional[TextInput] = None,
        dependencies: Iterable["ExprObject"] = [],
    ) -> None:
        if expr is not None and not isinstance(expr, sympy.Expr):
            raise TypeError(
                f"expr must be a string or sympy.Expr, got {type(expr).__name__}."
            )

        for dep in dependencies:
            if not isinstance(dep, ExprObject):
                raise TypeError(
                    f"dependencies must be LabeledExprObjects, got {type(dep).__name__}."
                )

        if expr is None and label is None:
            raise ValueError("Either expr or label must be specified.")

        self._expr = expr
        self._dependencies = tuple(dict.fromkeys(dependencies))  # Remove duplicates

        if expr is None and len(self._dependencies) > 0:
            raise ValueError("expr must be specified if object has dependencies.")

        self._label = Text.parse(label) if label is not None else None

    # @classmethod
    # def from_expr(
    #     cls, expr: sympy.Expr, dependencies: list["LabeledExprObject"]
    # ) -> Self:
    #     if not isinstance(expr, sympy.Expr):
    #         raise TypeError(f"expr must be a sympy.Expr, not {type(expr)}.")

    #     expr = sympy.simplify(expr)

    #     # Do not use the subclasses __init__ method, because this will probably be used to create
    #     # the object without an expression. Use __init_from_expr__ instead.
    #     obj: Self = object.__new__(cls, expr, dependencies)  # type: ignore
    #     ExprObject.__init__(obj, expr, dependencies)

    #     obj.__init_from_expr__()

    #     return obj
    @classmethod
    def labeled(cls, label: Text) -> Self:
        return cls(label=label)

    @classmethod
    def from_expr(cls, expr: sympy.Expr, dependencies: Iterable["ExprObject"]) -> Self:
        return cls(expr=expr, dependencies=dependencies)

    @property
    def expr(self) -> Optional[sympy.Expr]:
        return self._expr

    @property
    def dependencies(self) -> tuple["ExprObject", ...]:
        return self._dependencies

    @property
    def label(self) -> Optional[Text]:
        return self._label

    def ensure_has_expr(self):
        if self._expr is None:
            raise ValueError(
                f"{type(self).__name__} labeled '{self.label_or_fail().default}' has no expr."
            )

    def expr_or_fail(self):
        self.ensure_has_expr()
        return cast(sympy.Expr, self._expr)

    def ensure_has_label(self):
        if self._label is None:
            raise ValueError(
                f"{type(self).__name__} with expr '{self.expr_or_fail()}' has no label."
            )

    def label_or_fail(self):
        self.ensure_has_label()
        return cast(Text, self._label)

    @property
    def expr_text(self):
        self.ensure_has_expr()
        return Text(
            str(self._expr),
            unicode=ExprPrinter().doprint(self._expr),
            latex=sympy.latex(self._expr),
        )

    @property
    def text(self) -> Text:
        if self._label is not None:
            return self._label
        else:
            return self.expr_text

    @classmethod
    def create_eval_function(
        cls, expr: sympy.Expr, dependencies: Iterable["ExprObject"]
    ) -> Callable:

        # from sympy.utilities.lambdify import lambdastr

        # print(lambdastr([dep._expr for dep in self._dependencies], self._expr))

        return sympy.lambdify([dep._expr for dep in dependencies], expr)

    @classmethod
    def eval(cls, expr: sympy.Expr, dependency_values: dict["ExprObject", Any]):
        return cls.create_eval_function(expr, dependency_values.keys())(
            *dependency_values.values()
        )

    def __check_operation__(self, name: str, *args) -> None:
        pass

    def __is_one__(self) -> bool:
        return self._label is not None and self._label.default == "1"

    def __is_zero__(self) -> bool:
        return self._label is not None and self._label.default == "0"

    def __run_operation__(self, name: str, *args) -> Self:
        self.__check_operation__(name, *args)

        if len(self._dependencies) == 0:
            if isinstance(self, ExprObject):
                dependencies = [self]
            else:
                dependencies = []
        else:
            dependencies = list(self._dependencies)

        sympy_args = []
        for arg in args:
            if isinstance(arg, ExprObject):
                if arg.__is_zero__():
                    sympy_args.append(sympy.sympify(0))
                elif arg.__is_one__():
                    sympy_args.append(sympy.sympify(1))
                else:
                    sympy_args.append(arg._expr)

                    if len(arg._dependencies) == 0:
                        dependencies.append(arg)
                    for dep in arg._dependencies:
                        if dep not in dependencies:
                            dependencies.append(dep)
            else:
                sympy_args.append(sympy.sympify(arg))

        # print("argtypes:", [type(arg) for arg in sympy_args])

        expr = getattr(self._expr, name)(*sympy_args)

        if not isinstance(expr, sympy.Expr):
            raise TypeError(
                f"Operation {name}({', '.join(sympy_args)}) on {self._expr} returned {expr}, which is not a sympy.Expr."
            )

        return type(self).from_expr(expr, dependencies)

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
            if isinstance(self, ExprObject):
                dependencies = [self]
            else:
                dependencies = []
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

        if not isinstance(expr, sympy.Expr):
            raise TypeError(
                f"Array ufunc '{name}' {type(self).__name__} returned {expr}, which is not a sympy.Expr."
            )

        return type(self).from_expr(expr, dependencies)

    def __or__(self, label: TextInput):
        result = copy(self)
        # print("relabel", label, result)
        result._label = Text.parse(label)
        return result

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True

        if (
            not isinstance(other, ExprObject)
            or type(self) != type(other)
            or self._label != other._label
            or self._dependencies != other._dependencies
            or self._expr is None != other._expr is None
        ):
            return False

        if self._expr is not None:
            return sympy.simplify(self._expr - other._expr) == 0

        return True

    def __hash__(self) -> int:
        return hash((self._expr, self._dependencies))

    def __str__(self) -> str:
        return str(self._expr)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._expr}, dependencies={self._dependencies})"


# class LabeledExprObject(ExprObject):
#     _label: Text

#     def __init__(self, label: TextInput) -> None:
#         self.label = label
#         super().__init__(sympy.Symbol(self.label.default), [])

#     def __init_from_expr__(self):
#         self._label = super().text

#     @property
#     def label(self) -> Text:
#         return self._label

#     @label.setter
#     def label(self, label: TextInput) -> None:
#         self._label = Text.parse(label)
#         self._expr = sympy.Symbol(self._label.default)
#         # print("changed label to", self._label.default, "expr", self._expr)

#     def __or__(self, label: TextInput):
#         result = copy(self)
#         # print("relabel", label, result)
#         result.label = label
#         return result

#     @property
#     def text(self) -> Text:
#         return self._label

#     def __str__(self) -> str:
#         return self._label.default
