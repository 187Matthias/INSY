import numpy
from nn import NeuralNetwork
import os
# Immer im Ordner dieses Skripts arbeiten, damit "models/..." von überall aus gefunden wird
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from train_7seg import X, Y

net = NeuralNetwork.load("models/7seg.json")
richtig = 0
print("Geladenes 7-Segment-Netz:")
for d in range(10):
    bits = [int(round(v)) for v in net.predict(X[d])]
    richtig += bits == Y[d]
    print(f"  {d} {X[d]} -> {bits}  Soll {Y[d]}  {'OK' if bits == Y[d] else 'FALSCH'}")
print(f"{richtig}/10 Ziffern korrekt")

neu = NeuralNetwork.from_dict(net.to_dict())
print("Vorhersagen nach erneutem Export/Import identisch:",
      all(numpy.allclose(net.predict(x), neu.predict(x)) for x in X))
