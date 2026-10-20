import copy
import numpy
from nn import Layer

numpy.random.seed(2)
x = numpy.array([0.5, -1.0, 2.0])
ziel = numpy.array([1.0, 0.0])


def kosten(layer):
    return 0.5 * numpy.sum((layer.forward(x) - ziel) ** 2)


# 1) Form der Ausgabe
layer = Layer(2, 3)
out = layer.forward(x)
assert out.shape == (2,) and numpy.all((out > 0) & (out < 1))
print("1) forward: Form und Wertebereich OK ->", out)

# 2) Rückgegebener Fehler == numerischer Gradient dJ/dx (Gradient-Check)
vorher = copy.deepcopy(layer)
err_prev = layer.backward(out - ziel, 0.0)   # lr = 0: nichts wird verändert
num = numpy.zeros(3)
for i in range(3):
    xp = x.copy(); xp[i] += 1e-6
    xm = x.copy(); xm[i] -= 1e-6
    jp = 0.5 * numpy.sum((vorher.forward(xp) - ziel) ** 2)
    jm = 0.5 * numpy.sum((vorher.forward(xm) - ziel) ** 2)
    num[i] = (jp - jm) / 2e-6
assert numpy.allclose(err_prev, num, atol=1e-8)
print("2) Fehler an vorherige Schicht (Gradient-Check): OK")

# 3) Training über backward senkt den Fehler
layer = Layer(2, 3)
j0 = kosten(layer)
for _ in range(100):
    layer.backward(layer.forward(x) - ziel, 0.5)
j1 = kosten(layer)
assert j1 < j0
print(f"3) Training senkt Fehler: {j0:.6f} -> {j1:.6f}: OK")
print("Alle Tests bestanden.")
