import os
import numpy
from nn import NeuralNetwork

# Ziffer -> Segmente a b c d e f g (1 = leuchtet)
SEGMENTE = {
    0: [1, 1, 1, 1, 1, 1, 0],
    1: [0, 1, 1, 0, 0, 0, 0],
    2: [1, 1, 0, 1, 1, 0, 1],
    3: [1, 1, 1, 1, 0, 0, 1],
    4: [0, 1, 1, 0, 0, 1, 1],
    5: [1, 0, 1, 1, 0, 1, 1],
    6: [1, 0, 1, 1, 1, 1, 1],
    7: [1, 1, 1, 0, 0, 0, 0],
    8: [1, 1, 1, 1, 1, 1, 1],
    9: [1, 1, 1, 1, 0, 1, 1],
}

# Input: Ziffer als 4 Bit (MSB zuerst), z. B. 5 -> [0, 1, 0, 1]
X = [[(d >> b) & 1 for b in (3, 2, 1, 0)] for d in range(10)]
Y = [SEGMENTE[d] for d in range(10)]

if __name__ == "__main__":
    EPOCHS = 20000
    ALPHA = 0.1
    os.makedirs("models", exist_ok=True)

    numpy.random.seed(1)
    net = NeuralNetwork([4, 16, 7], learning_rate=ALPHA)
    net.fit(X, Y, EPOCHS, log_every=5000)

    print("\nErgebnis nach dem Training:")
    richtig = 0
    for d in range(10):
        p = net.predict(X[d])
        bits = [int(round(v)) for v in p]
        richtig += bits == Y[d]
        print(f"  {d} {X[d]} -> {bits}  Soll {Y[d]}  {'OK' if bits == Y[d] else 'FALSCH'}")
    print(f"{richtig}/10 Ziffern korrekt")

    net.dump_json("models/7seg.json")
    net.dump_txt("models/7seg.txt")
    net.dump_initfile("models/7seg_net.h", name="seg")
    print("\nGespeichert: models/7seg.json, 7seg.txt, 7seg_net.h")
