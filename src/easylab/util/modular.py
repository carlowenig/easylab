import textwrap
from typing import Any, Callable, Generic, TypeVar
from typing_extensions import Self

_T = TypeVar("_T")


class Modular:
    __modules: list[type] = []

    @classmethod
    def add_module(cls, module: type, *args, **kwargs):
        cls.__modules.append(module)

        for name, value in module.__dict__.items():
            if name in ["__module__", "__dict__", "__weakref__"]:
                continue
            elif name in ["__init__"]:
                raise ValueError(f"Cannot define attribute {name} in a module.")
            elif name == "__annotations__":
                cls.__annotations__.update(value)
            elif name == "__slots__":
                if not hasattr(cls, "__slots__"):
                    cls.__slots__ = []
                cls.__slots__.extend(value)
            elif name == "__doc__":
                module_doc = (
                    "[M] "
                    + module.__name__
                    + ": "
                    + textwrap.indent(value, " " * (len(module.__name__) + 6)).lstrip()
                )
                cls.__doc__ = (getattr(cls, "__doc__", "") or "") + "\n\n" + module_doc
            else:
                setattr(cls, name, value)

        last_arg_index = 0
        for name in module.__annotations__:
            if not hasattr(cls, name):
                if name in kwargs:
                    setattr(cls, name, kwargs[name])
                elif last_arg_index < len(args):
                    setattr(cls, name, args[last_arg_index])
                    last_arg_index += 1
                else:
                    raise ValueError(
                        f"Missing attribute '{name}' for {module.__name__}. Please provide it as an argument when calling add_module."
                    )

    @classmethod
    def get_modules(cls):
        return cls.__modules

    @classmethod
    def has_module(cls, module: type):
        return module in cls.__modules


class A(Modular):
    pass


class LabelModule:
    """A module which provides a label."""

    label: str

    def __str__(self) -> str:
        return self.label


A.add_module(LabelModule, label="Hello World!")
# print(A.__dict__)

print(A.label)
