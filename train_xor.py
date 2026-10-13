import os
import numpy
from nn import NeuralNetwork

# Trainingsdaten XOR
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]

EPOCHS = 20000
ALPHA = 0.1

os.makedirs("models", exist_ok=True)

# Mit nur 2 Hidden-Neuronen bleibt XOR bei ungünstigen Startgewichten manchmal
# in einem lokalen Minimum hängen. Dann starten wir mit neuen Zufallswerten neu.
for versuch in range(1, 11):
    numpy.random.seed(versuch)
    net = NeuralNetwork([2, 2, 1], learning_rate=ALPHA)
    cost = net.fit(X, Y, EPOCHS, log_every=5000)
    if all(abs(net.predict(x)[0] - y[0]) < 0.1 for x, y in zip(X, Y)):
        break
    print(f"Versuch {versuch}: nicht konvergiert (Kosten {cost:.4f}), neuer Start")

print("\nErgebnis nach dem Training:")
for x, y in zip(X, Y):
    print(f"  {x} -> {net.predict(x)[0]:.4f}  (Soll {y[0]})")

net.dump_json("models/xor.json")
net.dump_txt("models/xor.txt")
net.dump_initfile("models/xor_net.h", name="xor")
print("\nGespeichert: models/xor.json, xor.txt, xor_net.h")
