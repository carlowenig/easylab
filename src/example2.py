from matplotlib import pyplot as plt
import numpy as np
import easylab as lab

task = lab.Var(type=str)

I = lab.ValueVar(prec=2, unit=lab.units.A, fallback_err=0.1)
R = lab.ValueVar(prec=0, unit=lab.units.Hz)
X = lab.ValueVar()

Y = I * R ** 2

P520 = lab.Data()

totzeit = P520.select(task, "Totzeit")
totzeit.add_csv("src/totzeit.csv", (I, R))

print(P520.graph(I, R).info)

P520.graph(I, R).plot()
plt.show()

print(P520.select(I, 0.3).value(I * R).eq_text.default)

print((I * R).unit.dim)


print(lab.global_scope.tree())
