from dataclasses import dataclass
from typing import Union

import sympy
from ..lang import Text, TextInput

from ..util import LabeledExprObject


class Dim(LabeledExprObject):
    pass


# @dataclass(frozen=True)
# class Dimension_old:
#     time: int = 0
#     length: int = 0
#     mass: int = 0
#     temperature: int = 0
#     amount_of_substance: int = 0
#     current: int = 0
#     luminous_intensity: int = 0

#     @staticmethod
#     def parse(input: DimensionInput):
#         if input is None:
#             return Dimension()
#         elif isinstance(input, Dimension):
#             return input
#         elif isinstance(input, tuple):
#             return Dimension(*input)
#         elif isinstance(input, dict):
#             values = {}

#             for key in values:
#                 if key in _dimension_names:
#                     values[key] = input[key]
#                 elif key in _dimension_short_names:
#                     index = _dimension_short_names.index(key)
#                     values[_dimension_names[index]] = input[key]
#                 else:
#                     raise ValueError(f"Unknown dimension: {key}")

#             return Dimension(**input)
#         else:
#             raise TypeError(f"Cannot parse {input} as Dimension.")

#     def __mul__(self, other: "Dimension"):
#         return Dimension(
#             self.time + other.time,
#             self.length + other.length,
#             self.mass + other.mass,
#             self.temperature + other.temperature,
#             self.amount_of_substance + other.amount_of_substance,
#             self.current + other.current,
#             self.luminous_intensity + other.luminous_intensity,
#         )

#     def __pow__(self, exp: int):
#         return Dimension(
#             self.time * exp,
#             self.length * exp,
#             self.mass * exp,
#             self.temperature * exp,
#             self.amount_of_substance * exp,
#             self.current * exp,
#             self.luminous_intensity * exp,
#         )

#     def __truediv__(self, other: "Dimension"):
#         return self * other ** -1
