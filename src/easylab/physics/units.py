from functools import cache
from ..lang import lang
from ..util import Text
from . import dims
from .unit import Unit

# BASE UNITS
one = Unit("1")
"""Unit 1. Represents unitless values."""

m = metre = Unit(Text("m", long="metre", plural="metres"), dim=dims.length)
"""Metre (m) is the SI base unit of length."""

s = second = Unit(Text("s", long="second", plural="seconds"), dim=dims.time)
"""Second (s) is the SI base unit of time."""

g = gram = Unit(Text("g", long="gram", plural="grams"), dim=dims.mass, scale=1e-3)
"""Gram = g = 1/1000 kg"""

A = ampere = Unit(Text("A", long="ampere"), dim=dims.current)
"""Ampere (A) is the SI base unit of electric current."""

K = kelvin = Unit(Text("K", long="kelvin"), dim=dims.temperature)
"""Kelvin (K) is the SI base unit of temperature."""

mol = mole = Unit(
    Text("mol", long="mole", plural="moles"), dim=dims.amount_of_substance
)
"""Mole (mol) is the SI base unit of amount of substance."""

cd = candela = Unit(Text("cd", long="candela"), dim=dims.luminous_intensity)
"""Candela (cd) is the SI base unit of luminous intensity."""

# PREFIXES
yocto = Unit("y", scale=1e-24)
"""Prefix for 10^-24."""
zepto = Unit("z", scale=1e-21)
"""Prefix for 10^-21."""
atto = Unit("a", scale=1e-18)
"""Prefix for 10^-18."""
femto = Unit("f", scale=1e-15)
"""Prefix for 10^-15."""
pico = Unit("p", scale=1e-12)
"""Prefix for 10^-12."""
nano = Unit("n", scale=1e-9)
"""Prefix for 10^-9."""
micro = Unit(Text("mu", unicode="μ", latex="\\mu"), scale=1e-6)
"""Prefix for 10^-6."""
milli = Unit("m", scale=1e-3)
"""Prefix for 10^-3."""
centi = Unit("c", scale=1e-2)
"""Prefix for 10^-2."""
kilo = Unit("k", scale=1e3)
"""Prefix for 10^3."""
mega = Unit("M", scale=1e6)
"""Prefix for 10^6."""
giga = Unit("G", scale=1e9)
"""Prefix for 10^9."""
tera = Unit("T", scale=1e12)
"""Prefix for 10^12."""
peta = Unit("P", scale=1e15)
"""Prefix for 10^15."""
exa = Unit("E", scale=1e18)
"""Prefix for 10^18."""
zetta = Unit("Z", scale=1e21)
"""Prefix for 10^21."""
yotta = Unit("Y", scale=1e24)
"""Prefix for 10^24."""

# PREFIXED BASE UNITS
fm = femto % metre
"""Femtometre = fm = 10^-15 m."""
pm = pico % metre
"""Picometre = pm = 10^-12 m."""
nm = nano % metre
"""Nanometre = nm = 10^-9 m."""
um = micro % metre
"""Micrometre = um = 10^-6 m."""
mm = milli % metre
"""Millimetre = mm = 1/1000 m."""
cm = centi % metre
"""Centimetre = cm = 1/100 m."""
km = kilo % metre
"""Kilometre = km = 1000 m."""

fs = femto % second
"""Femtosecond = fs = 10^-15 s."""
ps = pico % second
"""Picosecond = ps = 10^-12 s."""
ns = nano % second
"""Nanosecond = ns = 10^-9 s."""
us = micro % second
"""Microsecond = us = 10^-6 s."""
ms = milli % second
"""Millisecond = ms = 1/1000 s."""

ug = micro % gram
"""Microgram = ug = 10^-9 kg."""
mg = milli % gram
"""Milligram = mg = 10^-6 kg."""
kg = kilo % gram
"""Kilogram (kg) is the SI base unit of mass."""

fA = femto % ampere
"""Femtoampere = fA = 10^-15 A."""
pA = pico % ampere
"""Picoampere = pA = 10^-12 A."""
nA = nano % ampere
"""Nanoampere = nA = 10^-9 A."""
uA = micro % ampere
"""Microampere = uA = 10^-6 A."""
mA = milli % ampere
"""Milliampere = mA = 1/1000 A."""
kA = kilo % ampere
"""Kiloampere = kA = 1000 A."""
MA = mega % ampere
"""Megaampere = MA = 10^6 A."""
GA = giga % ampere
"""Gigaampere = GA = 10^9 A."""


# DERIVED UNITS
mps = metres_per_second = m / s
"""Metres per second (m/s) is the SI derived unit of velocity."""
mps2 = metres_per_second_squared = m / s ** 2
"""Metres per second squared (m/s^2) is the SI derived unit of acceleration."""
m2 = square_metres = m ** 2
"""Square metres (m^2) is the SI derived unit of area."""
m3 = cubic_metres = m ** 3
"""Cubic metres (m^3) is the SI derived unit of volume."""
Hz = hertz = 1 / s | "Hz"
"""Hertz (Hz) is the SI derived unit of frequency."""
C = coulomb = A * s | "C"
"""Coulomb (C) is the SI derived unit of electric charge."""
N = newton = m * g * s ** 2 / m2 | "N"
"""Newton (N) is the SI derived unit of force."""
J = joule = kg * m ** 2 / s ** 2 | "J"
"""Joule (J) is the SI derived unit of energy."""
W = watt = J / s | "W"
"""Watt (W) is the SI derived unit of power."""
V = volt = W / A | "V"
"""Volt (V) is the SI derived unit of voltage or electric potential."""
ohm = V / A | lang.Omega
"""Ohm (Ω) is the SI derived unit of electric resistance."""
S = siemens = 1 / ohm | "S"
"""Siemens (S) is the SI derived unit of electric conductance."""
F = farad = C / V | "F"
"""Farad (F) is the SI derived unit of electric capacitance."""
H = henry = V * s / A | "H"
"""Henry (H) is the SI derived unit of electric inductance."""
T = tesla = V / m ** 2 | "T"
"""Tesla (T) is the SI derived unit of magnetic flux."""
Wb = weber = V * s | "Wb"
"""Weber (Wb) is the SI derived unit of magnetic flux density."""
lm = lumen = cd * m ** 2 | "lm"
"""Lumen (lm) is the SI derived unit of luminous flux."""
lx = lux = lm / m ** 2 | "lx"
"""Lux (lx) is the SI derived unit of illuminance."""

degC = degree_celsius = K | "degC"
"""Degree Celsius (°C) is a unit of temperature."""
degC.offset = 273.15


# Helper functions
def _collect_units_with_names():
    """Collect all units with names."""
    units_with_names: dict[Unit, list[str]] = {}

    for name, value in globals().items():
        if isinstance(value, Unit):
            if value not in units_with_names:
                units_with_names[value] = []
            units_with_names[value].append(name)

    return units_with_names


all_with_names = _collect_units_with_names()
all = list(all_with_names.keys())


def _collect_units_by_query_strings():
    results: dict[str, Unit] = {}
    for unit, names in all_with_names.items():
        for name in names:
            results[name] = unit
        for query_string in unit.label.query_strings:
            results[query_string] = unit
    return results


all_by_query_strings = _collect_units_by_query_strings()


def find_or_none(query: str):
    return all_by_query_strings.get(query)


def find(query: str):
    result = find_or_none(query)
    if result is None:
        raise ValueError(f"No unit found for query '{query}'.")
    return result
