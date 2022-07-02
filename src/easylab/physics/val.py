from typing import Optional

from .dim import Dim
from .unit import Unit
from . import units


class Val:
    _mean: float
    _err: float
    _dim: Dim

    def __init__(self, mean: float, err: float, dim: Dim) -> None:
        self._mean = mean
        self._err = err
        self._dim = dim

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def err(self) -> float:
        return self._err

    @property
    def dim(self) -> Dim:
        return self._dim

    def rep(self, *, precision: Optional[int] = None, unit: Unit = units.one):
        return ValRep(self, precision=precision, unit=unit)


class ValRep:
    _val: Val
    precision: Optional[int]
    _unit: Unit

    def __init__(
        self, val: Val, *, precision: Optional[int] = None, unit: Unit = units.one
    ) -> None:
        self._val = val
        self.precision = precision
        self._unit = unit
        self._check_unit()

    @property
    def val(self) -> Val:
        return self._val

    @property
    def unit(self) -> Unit:
        return self._unit

    @unit.setter
    def unit(self, unit: Unit) -> None:
        self._unit = unit
        self._check_unit()

    def _check_unit(self):
        if self._unit.dim != self._val.dim:
            raise ValueError(
                f"Unit must be of dimension {self._val.dim}, got {self.unit.dim}."
            )
