# from abc import ABC, abstractmethod
# from typing import Any, Callable, Optional, Union, overload, TypeVar

# from ..lang import is_text_input, Text, TextInput
# from ..data import Var, OutputTarget

# _T = TypeVar("_T")


# class Job(ABC):
#     _args: tuple
#     _kwargs: dict[str, Any]
#     _vars: list[Var]

#     def __init__(self, *args, **kwargs) -> None:
#         self._args = args
#         self._kwargs = kwargs
#         self.setup()

#     def setup(self):
#         pass

#     @abstractmethod
#     def run(self):
#         pass

#     def teardown(self):
#         pass

#     @overload
#     def var(self, var: Var[_T], /) -> Var[_T]:
#         ...

#     @overload
#     def var(
#         self,
#         label: TextInput,
#         /,
#         *,
#         default: Optional[_T] = None,
#         type: type[_T] = Any,
#         format: Optional[Callable[[_T], TextInput]] = None,
#         parse: Optional[Callable[[Any], _T]] = None,
#         check: Optional[Callable[[_T], Union[bool, str, None]]] = None,
#         output: Optional[Callable[[_T, OutputTarget], Any]] = None,
#     ) -> Var[_T]:
#         ...

#     def var(self, arg, /, **kwargs) -> Var:
#         if isinstance(arg, Var) and len(kwargs) == 0:
#             var = arg
#         elif is_text_input(arg):
#             label = Text.parse(arg)
#             for v in self._vars:
#                 if v.label == label:
#                     var = v
#                     break
#             else:
#                 var = Var(label, **kwargs)
#         else:
#             raise TypeError(f"Cannot create Var from argument {arg}.")

#         if var not in self._vars:
#             self._vars.append(var)
#         return var

#     def arg(self, index: int) -> Any:
#         return self._args[index]

#     def kwarg(self, name: str) -> Any:
#         if name not in self._kwargs:
#             raise KeyError(f"No kwarg named {name}.")
#         return self._kwargs[name]


# x = Var("x")
# y = Var("y")


# @job
# def create_linear_fit(x: Var[float], y: Var[float], source: str):


# create_linear_fit(x=1, y=2)
