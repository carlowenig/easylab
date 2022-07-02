import sympy
from .dim import Dim

# time = Dimension(time=1)
# length = Dimension(length=1)
# mass = Dimension(mass=1)
# temperature = Dimension(temperature=1)
# amount_of_substance = Dimension(amount_of_substance=1)
# current = Dimension(current=1)
# luminous_intensity = Dimension(luminous_intensity=1)

# velocity = length / time
# acceleration = velocity / time
# force = mass * acceleration

# energy = force * length
# power = energy / time
# voltage = power / current
# charge = time * current


one = Dim.from_expr(sympy.Number(1), [])

time = Dim("T")
length = Dim("L")
mass = Dim("M")
temperature = Dim("Θ")
amount_of_substance = Dim("N")
current = Dim("I")
luminous_intensity = Dim("J")

velocity = length / time
acceleration = velocity / time
force = mass * acceleration

energy = force * length
power = energy / time
voltage = power / current
charge = time * current
