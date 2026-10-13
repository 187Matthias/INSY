import numpy


def sigmoid(x):
    # Sigmoid quetscht jeden Wert in den Bereich (0, 1): 1 / (1 + e^-x)
    # (Im alten Code stand numpy.e^-x - "^" ist in Python XOR, nicht Potenz!)
    return 1 / (1 + numpy.exp(-x))


def sigmoid_derivative(a):
    # Ableitung der Sigmoid-Funktion, ausgedrückt durch die AUSGABE a = sigmoid(z):
    # sigmoid'(z) = a * (1 - a)
    # Deshalb bekommt diese Funktion den Output und nicht das z.
    return a * (1 - a)


class Neuron:
    """Ein einzelnes Neuron: output = activation(dot(inputs, weights) + bias)"""

    def __init__(self, num_inputs, activation_function=sigmoid,
                 activation_function_d=sigmoid_derivative):
        # Zufällige Startwerte in [-1, 1]. Wären alle Gewichte gleich (z. B. 0),
        # würden alle Neuronen einer Schicht das Gleiche lernen.
        self.weights = numpy.random.uniform(-1, 1, size=num_inputs)
        self.bias = numpy.random.uniform(-1, 1)
        self.activation_function = activation_function
        self.activation_function_d = activation_function_d
        # Letzte Inputs/Output merken - werden im Backward-Pass gebraucht
        self.inputs = None
        self.output = None

    def activation(self, inputs):
        """Forward-Pass: z = w·x + b, danach Aktivierungsfunktion."""
        self.inputs = numpy.asarray(inputs, dtype=float)
        z = numpy.dot(self.inputs, self.weights) + self.bias
        self.output = self.activation_function(z)
        return self.output

    def activation_d(self):
        """Ableitung der Aktivierung an der letzten Stelle (dA/dZ)."""
        return self.activation_function_d(self.output)

    def update(self, delta, learning_rate):
        """Gradient-Descent-Schritt: W_neu = W_alt - alpha * dJ/dW

        delta = dJ/dZ (Fehler * Aktivierungs-Ableitung, vom Layer berechnet).
        Kettenregel:  dJ/dW = dJ/dZ * dZ/dW = delta * inputs
                      dJ/db = dJ/dZ * dZ/db = delta * 1
        """
        self.weights -= learning_rate * delta * self.inputs
        self.bias -= learning_rate * delta
