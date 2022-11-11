from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Union, cast
from typing_extensions import Self
import sympy
from varname import varname, ImproperUseError
from ..lang import Text

# from ..physics import Unit
# from ..lang import Text


VarIdentifier = Union[str, sympy.Symbol]


def var_name(identifier: VarIdentifier):
    if isinstance(identifier, str):
        return identifier
    elif isinstance(identifier, sympy.Symbol):
        return identifier.name
    else:
        raise ValueError(f"Invalid var identifier {identifier}.")


def _infer_scope_name():
    try:
        return cast(str, varname(2))
    except ImproperUseError:
        return "unknown"
        # raise ValueError(
        #     f"Could not infer scope name. Please specify one or directly assign the scope to a variable to use its name."
        # )


class CancelScopeAction(Exception):
    pass


class Module:
    def __init__(self, next: Module) -> None:
        self.next = next

    @property
    def last(self):
        if self.next is self:
            return self
        else:
            return self.next.last

    def on_set_prop(self, var: VarIdentifier, prop: str, value: Any):
        self.next.on_set_prop(var, prop, value)

    def on_define(self, name: str, props: dict[str, Any]):
        self.next.on_define(name, props)

    @property
    def known_props(self) -> set[str]:
        return set()


class PropModule(Module):
    prop: str

    def on_set(self, var: VarIdentifier, value: Any) -> None:
        self.next.on_set_prop(var, self.prop, value)

    def on_set_prop(self, var: VarIdentifier, prop: str, value: Any):
        if prop == self.prop:
            self.on_set(var, value)
        else:
            super().on_set_prop(var, prop, value)

    @property
    def known_props(self) -> set[str]:
        return {self.prop}


class EndModule(Module):
    def __init__(self, space: Space) -> None:
        self.space = space
        super().__init__(self)

    def on_set_prop(self, var: VarIdentifier, prop: str, value: Any):
        self.space.get_props(var)[prop] = value

    def on_define(self, name: str, props: dict[str, Any]):
        self.space.vars[name] = {}

        for prop, value in props.items():
            self.space.set(name, prop, value)


class Parsable(Protocol):
    @classmethod
    @abstractmethod
    def parse(cls, input: Any) -> Self:
        pass


class PropParseModule(PropModule):
    type: type[Parsable]

    def on_set(self, var: VarIdentifier, value: Any):
        super().on_set(var, self.type.parse(value))


class VarSymbol(sympy.Symbol):
    space: Space

    @property
    def props(self):
        return self.space.get_props(self)

    def __getattr__(self, name: str):
        if name in self.props:
            return self.props[name]
        else:
            return super().__Symbol_getattr__(name)  # type: ignore


@dataclass
class Space:
    name: str = field(default_factory=_infer_scope_name)
    vars: dict[str, dict[str, Any]] = field(default_factory=dict)
    modules: list[Module] = field(default_factory=list)
    # prop_parsers: dict[str, list[Callable[[Any], Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.modules.append(EndModule(self))

    def has(self, var: VarIdentifier):
        return var_name(var) in self.vars

    def get_props(self, var: VarIdentifier):
        name = var_name(var)
        if name not in self.vars:
            raise ValueError(f"Var {name} is not defined.")
        return self.vars[name]

    def get(self, var: VarIdentifier, prop: str):
        return self.get_props(var).get(prop, None)

    def set(self, var: VarIdentifier, prop: str, value: Any):
        self.start_module.on_set_prop(var, prop, value)

    def create_var(self, name: str | None = None, **props):
        if name is None:
            name = cast(str, varname(2))

        if name in self.vars:
            raise ValueError(f"Var {name} is already defined.")

        self.start_module.on_define(name, props)
        return self.get_var(name)

    def get_var(self, name: str):
        if name not in self.vars:
            raise ValueError(f"Var {name} is not defined.")

        symbol = VarSymbol(name)
        symbol.space = self
        return symbol

    def var(self, name: str | None = None, **props):
        if len(props) == 0 and name in self.vars:
            return self.get_var(name)
        else:
            return self.create_var(name, **props)

    def use(self, *module_factories: Callable[[Module], Module]):
        for module_factory in module_factories:
            next = self.modules[-1]
            module = module_factory(next)
            if not isinstance(module, Module):
                raise ValueError(
                    f"Module factory did not return a Module. Got {type(module)}."
                )
            self.modules.append(module)

    @property
    def start_module(self):
        return self.modules[-1]

    @property
    def known_props(self):
        return set.union(*(module.known_props for module in self.modules))

    def __getattr__(self, name: str):
        if name.startswith("set_"):
            prop = name[4:]
            return lambda var, value=None: cast(Any, self.set(var, prop, value))
        elif name.startswith("get_"):
            prop = name[4:]
            return lambda var, value=None: cast(Any, self.get(var, prop))
        elif self.has(name):
            return self.get_var(name)
        else:
            raise AttributeError(f"Space has no attribute '{name}'.")

    def __str__(self) -> str:
        return f"Space({self.name}, {len(self.vars)} vars, {len(self.modules) - 1} modules)"  # Ignore end module

    def __dir__(self):
        keys = list(super().__dir__())

        for prop in self.known_props:
            keys.append(f"get_{prop}")
            keys.append(f"set_{prop}")

        keys.extend(self.vars.keys())

        return keys

    # def __enter__(self):
    #     set_scope(self)

    # def __exit__(self, type_, value, traceback):
    #     pop_scope(self)

    # def var(self, *args, **kwargs):
    #     if len(args) == 0:
    #         name = cast(str, varname())
    #         type_ = None
    #     elif len(args) == 1:
    #         (arg,) = args
    #         if isinstance(arg, str):
    #             name = arg
    #             type_ = None
    #         elif isinstance(arg, type):
    #             name = cast(str, varname())
    #             type_ = arg
    #         else:
    #             raise ValueError(f"Invalid first argument {arg}.")
    #     else:
    #         raise ValueError(f"Expected 0-1 positional args. Got {len(args)}.")

    #     if name in self.var_types:
    #         raise ValueError(f"Var {name} is already defined.")

    #     self.var_types[name]

    #     symbol = sympy.Symbol(name)

    #     return symbol


# __default_scope = Scope()
# __scope_stack = [__default_scope]


# def set_scope(scope: Scope | None = None):
#     if scope is None:
#         scope = Scope()

#     __scope_stack.append(scope)

#     return scope

# def pop_scope(scope: Scope | None = None):
#     if scope is not None and get_scope() != scope:
#         raise ValueError(f"Cannot pop scope {scope}, since it is not the current scope.")
#     return __scope_stack.pop()

# def reset_scope():
#     __scope_stack = [__default_scope]


# def get_scope():
#     return __scope_stack[-1]


# def define(name: str | None = None, **props):
#     return get_scope().define(name, **props)

# TODO: Use real unit parsing
@dataclass
class Unit:
    name: str

    @classmethod
    def parse(cls, input: Any):
        if isinstance(input, Unit):
            return input
        elif isinstance(input, str):
            return Unit(input)
        else:
            raise ValueError(f"Cannot parse Unit from {input}.")


class UnitModule(PropParseModule):
    prop = "unit"
    type = Unit

    def on_define(self, name: str, props: dict[str, Any]):
        if "unit" not in props:
            if "expr" in props:
                expr: sympy.Expr = props["expr"]
                args = list(expr.free_symbols)
                f = sympy.lambdify(args, props["expr"])
                unit = f(*(arg.unit for arg in args))
                props |= {"unit": unit}

            else:
                props |= {"unit": Unit("1")}

        super().on_define(name, props)


class LabelModule(Module):
    def on_define(self, name: str, props: dict[str, Any]):
        if "label" not in props:
            props |= {"label": Text(name)}  # TODO: Use Text(name)
        else:
            props["label"] = Text.parse(props["label"])
        super().on_define(name, props)

    @property
    def known_props(self) -> set[str]:
        return {"label"}


dims = Space()
dims.use(LabelModule)
dims.var("L", label=Text("L", long="length"))



units = Space()
units.use(LabelModule, LabelModule)
units.var("m", dim=dims.L, label=Text("m", long="metre"))
print(units.m)

# lab = Lab()
# lab.use(UnitModule, LabelModule)


# x = lab.var(unit="m", prec=3)
# t = lab.var(unit="s", prec=2)

# a = lab.var(expr=x / t ** 2)

# print(a.unit)
