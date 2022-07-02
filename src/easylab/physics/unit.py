from cProfile import label
from typing import Optional
import sympy


from . import dims
from .dim import Dim
from ..lang import Text, TextInput, lang
from ..util import LabeledExprObject


class Unit(LabeledExprObject):
    dim: Dim
    scale: float
    offset: float

    def __init__(
        self,
        label: TextInput,
        dim: Dim = dims.one,
        *,
        scale: float = 1,
        offset: float = 0,
    ) -> None:

        self.dim = dim
        self.scale = scale
        self.offset = offset

        super().__init__(label)

    def __init_from_expr__(self):
        super().__init_from_expr__()

        f = self.create_eval_function()

        unit_deps: list[Unit] = []
        for dep in self.dependencies:
            if isinstance(dep, Unit):
                if dep.offset != 0:
                    raise ValueError("Cannot compose units with offsets.")
                unit_deps.append(dep)
            else:
                raise ValueError(
                    f"Units cannot only be composed of other units. Got {type(dep)}."
                )

        self.dim = f(*(dep.dim for dep in unit_deps))
        self.scale = f(*(dep.scale for dep in unit_deps))
        self.offset = 0  # TODO

    @property
    def label(self) -> Text:
        return super().label

    @label.setter
    def label(self, label: TextInput):
        self._label = lang.mathrm(Text.parse(label))

    def prefix(self, prefix_unit: "Unit"):
        if self.label is None or prefix_unit.label is None:
            raise ValueError("Cannot prefix units without labels.")

        result = prefix_unit * self
        result.label = prefix_unit.label + self.label
        return result

    def __mod__(self, other: "Unit") -> "Unit":
        return other.prefix(self)

    def convert(self, value: float, to: "Unit"):
        if self.dim != to.dim:
            raise ValueError(f"Cannot convert from unit {self} to {to}.")

        return (value - self.offset) * self.scale / to.scale + to.offset
