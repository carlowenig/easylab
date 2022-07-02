from glob import glob
import sys
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from . import constraint


from ..lang import TextInput, Text
from ..util import LabeledExprObject, AutoNamed


_T = TypeVar("_T")


class Scope(AutoNamed):
    _parent: "Scope"
    _vars: list["Var"]
    _children: list["Scope"]

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        parent: Optional["Scope"] = None,
    ) -> None:
        self.__init_auto_named__(name)

        if parent != "__UNSET__":
            self._parent = parent or current_scope
            self._parent._children.append(self)

        self._vars = []
        self._children = []

    @property
    def vars(self) -> list["Var"]:
        return list(self._vars)

    @property
    def children(self) -> list["Scope"]:
        return list(self._children)

    @property
    def name(self) -> str:
        return self._name

    @property
    def parent(self) -> "Scope":
        return self._parent

    @property
    def ancestors(self) -> list["Scope"]:
        if self.is_global:
            return []
        else:
            return [self.parent] + self.parent.ancestors

    @property
    def is_global(self):
        return self is global_scope

    def is_child_of(self, other: "Scope"):
        return other in self.ancestors

    def _add(self, var: "Var"):
        if var in self._vars:
            return

        if any(v.name == var.name for v in self._vars):
            raise RuntimeError(
                f"There already exists a variable named '{var.name}' in scope {self.name}."
            )

        self._vars.append(var)
        self.__dict__[var.name] = var

    def index(self, query: Any) -> Optional[int]:
        for i, var in enumerate(self._vars):
            if var == query or var.name == query:
                return i

    def get(self, query: Any, *, ancestors: bool = True) -> Optional["Var"]:
        index = self.index(query)

        if index is not None:
            return self._vars[index]

        if ancestors and self._parent is not None:
            return self._parent.get(query, ancestors=True)

    def __getitem__(self, query: Any) -> Optional["Var"]:
        return self.get(query)

    def __contains__(self, query: Any) -> bool:
        return self.get(query) is not None

    def __enter__(self):
        if current_scope is not self._parent:
            raise RuntimeError(
                f"Cannot enter scope {self.name} (child of {self.parent.name}) because the current scope is {current_scope.name}."
            )
        set_current_scope(self)

    def __exit__(self, type, value, traceback):
        set_current_scope(self.parent)

    def tree(self, *, include_hidden_vars: bool = False) -> str:
        return (
            self.name
            + "".join(
                f"\n| {v.name}: {v.type}"
                for v in self.vars
                if include_hidden_vars or not v.is_hidden
            )
            + "".join(
                "\n|\no-- " + s.tree.replace("\n", "\n|   ") for s in self.children
            )
        )

    def __str__(self) -> str:
        return self.name

    def __repr__(self):
        return f"<Scope {self.name}>"


global_scope = Scope(name="global", parent=cast(Scope, "__UNSET__"))
global_scope._parent = global_scope

current_scope = global_scope


def set_current_scope(scope: Scope):
    global current_scope
    current_scope = scope


def pop_current_scope():
    set_current_scope(current_scope.parent)


def get_common_scope(*scopes: Scope):
    if len(scopes) == 0:
        return global_scope

    common = scopes[0]
    if common.is_global:
        return global_scope

    for scope in scopes[1:]:
        if scope.is_global:
            return global_scope

        if scope != common:
            for ancestor in scope.ancestors:
                if ancestor in common.ancestors:
                    common = ancestor
                    break
    return common


def handle_check_result(result: Any):
    if isinstance(result, str):
        raise ValueError(result)
    elif result is False:
        raise ValueError("Invalid input.")


OutputTarget = Literal["plot", "plot_err"]


class Var(LabeledExprObject, Generic[_T], AutoNamed):
    name: str
    default: Optional[_T]
    type: Type[_T]
    _format_func: Optional[Callable[[_T], TextInput]]
    _parse_func: Optional[Callable[[Any], _T]]
    _check_func: Optional[Callable[[_T], Union[bool, str, None]]]
    _output_func: Optional[Callable[[_T, OutputTarget], Any]]
    _scope: Scope

    def __init__(
        self,
        label: Optional[TextInput] = None,
        *,
        default: Optional[_T] = None,
        type: Type[_T] = Any,
        format: Optional[Callable[[_T], TextInput]] = None,
        parse: Optional[Callable[[Any], _T]] = None,
        check: Optional[Callable[[_T], Union[bool, str, None]]] = None,
        output: Optional[Callable[[_T, OutputTarget], Any]] = None,
        scope: Optional[Scope] = None,
        name: Optional[str] = None,
        auto_name: bool = True,
    ):

        self.default = default
        self.type = type
        self._format_func = format
        self._parse_func = parse
        self._check_func = check
        self._output_func = output
        self._scope = scope or current_scope

        self.__init_auto_named__(
            name,
            fallback=lambda: Text.parse(label).default if label is not None else None,
            auto_name=auto_name,
        )

        if label is None:
            label = self.name

        super().__init__(label)

        self._scope._add(self)

    def __init_from_expr__(self):
        super().__init_from_expr__()

        found_name = [
            True
        ]  # Use list to be able to change value from inside fallback function

        def fallback():
            found_name[0] = False
            return self.label.default

        self.__init_auto_named__(
            fallback=fallback,
            find_name_depth_offset=3,  # This function is called from three other functions
        )
        if found_name[0]:
            self.label = self.name

        f = self.create_eval_function()

        var_deps: list[Var] = []
        for dep in self.dependencies:
            if isinstance(dep, Var):
                var_deps.append(dep)
            else:
                raise ValueError(
                    f"Vars can only be composed of other vars. Got {type(dep)}."
                )

        dep_defaults = [dep.default for dep in var_deps]

        self.default = (
            f(dep_defaults) if all(d is not None for d in dep_defaults) else None
        )
        self.type = Any
        self._format_func = None
        self._parse_func = None
        self._check_func = None
        self._scope = get_common_scope(*(dep.scope for dep in var_deps))

    @property
    def is_computed(self):
        return len(self.dependencies) > 0

    @property
    def is_hidden(self):
        return self.name.startswith("__")

    @property
    def var_dependencies(self):
        return cast(list[Var], self.dependencies)

    @property
    def scope(self):
        return self._scope

    def is_in_scope(self, scope: Scope, *, include_ancestors: bool = True) -> bool:
        if self.scope is scope:
            return True
        if include_ancestors:
            for ancestor in self.scope.ancestors:
                if self.scope == ancestor:
                    return True
        return False

    def __lt__(self, other: Any) -> "constraint.BoundsConstraint[_T]":  # type: ignore
        return constraint.BoundsConstraint(self, max=other, include_max=False)  # type: ignore

    def __lte__(self, other: Any) -> "constraint.BoundsConstraint[_T]":  # type: ignore
        return constraint.BoundsConstraint(self, max=other, include_max=True)  # type: ignore

    def __gt__(self, other: Any) -> "constraint.BoundsConstraint[_T]":  # type: ignore
        return constraint.BoundsConstraint(self, min=other, include_min=False)  # type: ignore

    def __gte__(self, other: Any) -> "constraint.BoundsConstraint[_T]":  # type: ignore
        return constraint.BoundsConstraint(self, min=other, include_min=True)  # type: ignore

    def __mod__(self, other: Any):
        return constraint.EqualConstraint(self, other)

    def format(
        self,
        input: Any,
        *,
        parse: bool = True,
        check: bool = True,
        check_type: bool = True,
    ) -> Text:
        if parse:
            value = self.parse(input, check=check, check_type=check_type)
        else:
            value = cast(_T, input)
            if check:
                self.check(value, check_type=check_type)

        if self._format_func is not None:
            return Text.parse(self._format_func(value))
        else:
            return Text.parse(self._format_fallback(value))

    def parse(self, input: Any, *, check: bool = True, check_type: bool = True) -> _T:
        if self._parse_func is not None:
            value = self._parse_func(input)
        else:
            value = self._parse_fallback(input)

        if check:
            try:
                self.check(value, check_type=check_type)
            except Exception as e:
                raise ValueError(
                    f"Got invalid value {value} for variable {self} when parsing {input}. Value Check failed because: {e}"
                )

        return value

    def check(self, value: _T, *, check_type: bool = True):
        if check_type:
            if not self.type is Any and not isinstance(value, self.type):
                raise ValueError(
                    f"Invalid type. Expected {self.type.__name__}, got {type(value).__name__}."
                )

        if self._check_func is not None:
            handle_check_result(self._check_func(value))

        handle_check_result(self._check(value))

    def output(
        self,
        input: Any,
        target: OutputTarget,
        *,
        parse: bool = True,
        check: bool = True,
        check_type: bool = True,
    ):
        if parse:
            value = self.parse(input, check=check, check_type=check_type)
        else:
            value = cast(_T, input)
            if check:
                self.check(value, check_type=check_type)

        if self._output_func is not None:
            return self._output_func(value, target)
        else:
            return self._output_fallback(value, target)

    def _parse_fallback(self, input: Any) -> _T:
        if isinstance(input, self.type):
            return input
        elif hasattr(self.type, "parse"):
            return getattr(self.type, "parse")(input)
        elif callable(self.type):
            try:
                return self.type(input)
            except:
                return input
        else:
            return input

    def _format_fallback(self, value: _T) -> TextInput:
        if hasattr(value, "text"):
            return getattr(value, "text")
        else:
            return str(value)

    def _output_fallback(self, value: _T, target: OutputTarget):
        return value

    def _check(self, value: _T):
        pass
