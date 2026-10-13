import numpy
from nn import NeuralNetwork

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]

net = NeuralNetwork.load("models/xor.json")
ok = True
print("Geladenes XOR-Netz:")
for x, y in zip(X, Y):
    p = net.predict(x)[0]
    print(f"  {x} -> {p:.4f}  (Soll {y[0]}, gerundet {round(p)})")
    ok &= round(p) == y[0]

# Export -> Import darf die Vorhersagen nicht verändern
neu = NeuralNetwork.from_dict(net.to_dict())
gleich = all(numpy.allclose(net.predict(x), neu.predict(x)) for x in X)
print("Vorhersagen nach erneutem Export/Import identisch:", gleich)
print("XOR korrekt:", ok)
