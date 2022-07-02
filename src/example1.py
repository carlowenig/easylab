from typing import Any, Optional

import numpy as np
import easylab as lab
import lmfit
from easylab.data.measured import Measured, Unit


class RoughFloat(lab.Var[float]):
    type = float
    precision: int

    def __init__(self, name: str, *, precision: int = -3, **kwargs):
        super().__init__(name, **{"default": 0, **kwargs})
        self.precision = precision

    def format(self, value: float):
        if self.precision < 0:
            return f"{value:.{-self.precision}f}"
        else:
            return str(int(value))

    def equal(self, a: float, b: float):
        return abs(a - b) < 10 ** self.precision

    def merge(self, a: float, b: float):
        if self.equal(a, b):
            return (a + b) / 2

        return super().merge(a, b)


class MeasuredVar(lab.Var[lab.Measured]):
    type = lab.Measured
    precision: int
    unit: Optional[Unit]

    def __init__(
        self, name: str, *, precision: int = -3, unit: Optional[Unit] = None, **kwargs
    ):
        super().__init__(name, **{"default": lab.Measured(0), **kwargs})
        self.precision = precision
        self.unit = unit

    def equal(self, a: lab.Measured, b: lab.Measured):
        return (
            abs(a.value - b.value) < 10 ** self.precision
            and abs(a.error - b.error) < 10 ** self.precision
            and a.unit == b.unit
        )

    def parse(self, input: Any, *, check: bool = True, use_default: bool = True):
        if use_default and input is None:
            return self.default
        return Measured.parse(input, unit=self.unit)

    def merge(self, a: lab.Measured, b: lab.Measured):
        if self.equal(a, b):
            return lab.Measured(
                (a.value + b.value) / 2, error=(a.error + b.error) / 2, unit=a.unit
            )

        return super().merge(a, b)

    def check(self, value: Measured, *, check_type: bool = True):
        super().check(value, check_type=check_type)

        if self.unit is not None and value.unit != self.unit:
            raise ValueError(f"Expected unit {self.unit}, got {value.unit}.")


class Id(lab.Var[str]):
    type = str

    def format(self, value: str):
        return f'"{value}"'

    def merge(self, a: str, b: str):
        return f"{a}, {b}"


id = Id("id")

t = MeasuredVar("t", unit=Unit("s"))
x = MeasuredVar("x", precision=-1, unit=Unit("m"))
y = MeasuredVar("y", unit=Unit("m"))
z = MeasuredVar("y", unit=Unit("m"))

series = lab.Series(
    {z: 5},  # z is 5 independently of the other variables
    {t: 0, x: 0, id: "M1"},
    {t: 1, x: 5.001, id: "M2"},
    {x: 5, y: 3, id: "M3"},
    {x: 0, y: 4, id: "M3a"},
)
print(series.where(x=5).get(z))
# print(coll.where(x, 5).set(y, 6))

print(series.find(x, 5))


data = lab.Data()
exercise = lab.Var("exercise", type=str)
U = lab.Var("U", type=float)
Q = lab.Var("Q", type=float)
C = lab.Var("C", type=float)


capacity_measurements = data.where({exercise: "Messung der Kapazitäten"})

# Different ways to add new data
capacity_measurements.add(
    {U: 1, Q: 1},
    {U: 2, Q: 4},
    {U: 3, Q: 6},
)

capacity_measurements.add_rows(
    [U, Q],
    [1, 1],
    [2, 4],
    [3, 6],  #
)

capacity_measurements.add_columns(
    [U, 1, 2, 3],
    [Q, 1, 4, 6],  #
)

capacity_measurements.add_csv("capacities.csv")


# This will fit the linear model Q / C to the graph (U, Q).
# The function should detect, that there are values given for Q und C is a free parameter.
# The result values for C will be saved in the series.
capacity_measurements.fit(U, Q / C)

# After this, we can get the fitted values for C.
capacity_measurements.get(C)


# a = lab.Precise("5.680+")
# print("a =", repr(a))
# b = lab.Precise(0.01)
# print("b =", repr(b))
# sum = a + b
# print("sum =", repr(sum))
# cos = np.cos(sum)
# print("cos =", repr(cos))
