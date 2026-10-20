import numpy
from .neuron import Neuron, sigmoid, sigmoid_derivative


class Layer:
    """Eine Schicht: Liste von Neuronen, die alle die gleichen Inputs bekommen."""

    def __init__(self, num_neurons, num_inputs, activation_function=sigmoid,
                 activation_function_d=sigmoid_derivative):
        self.neurons = [Neuron(num_inputs, activation_function, activation_function_d)
                        for _ in range(num_neurons)]

    def forward(self, inputs):
        """Forward-Pass: jedes Neuron berechnet seinen Output, Ergebnis = Vektor."""
        return numpy.array([neuron.activation(inputs) for neuron in self.neurons])

    def backward(self, errors, learning_rate):
        """Backward-Pass der Schicht.

        errors[k] = dJ/dA_k, also wie stark der Output von Neuron k den Fehler J
        beeinflusst. Für die letzte Schicht ist das (y_hat - y) (Ableitung von
        J = 1/2 (y_hat - y)^2), für alle anderen kommt es von der nächsten Schicht.

        Rückgabe: Fehler für die vorherige Schicht, dJ/dA_prev.
        """
        # Kettenregel: dJ/dZ_k = dJ/dA_k * dA_k/dZ_k  (= errors * a*(1-a))
        deltas = [error * neuron.activation_d()
                  for error, neuron in zip(errors, self.neurons)]

        # dJ/dA_prev = Summe über alle Neuronen k von dJ/dZ_k * dZ_k/dA_prev
        # und dZ_k/dA_prev = w_k. Das MUSS mit den alten Gewichten passieren,
        # also bevor update() sie verändert!
        errors_prev = numpy.zeros_like(self.neurons[0].weights)
        for delta, neuron in zip(deltas, self.neurons):
            errors_prev += delta * neuron.weights

        # Erst jetzt Gewichte und Bias aktualisieren
        for delta, neuron in zip(deltas, self.neurons):
            neuron.update(delta, learning_rate)

        return errors_prev
