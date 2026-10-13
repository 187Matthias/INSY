import json
import numpy
from .layer import Layer


class NeuralNetwork:
    """Ein vollständig verbundenes Netz aus mehreren Layern.

    layer_sizes = [2, 2, 1] bedeutet: 2 Inputs, 2 Hidden-Neuronen, 1 Output-Neuron.
    """

    def __init__(self, layer_sizes, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.layer_sizes = list(layer_sizes)
        # Layer i hat layer_sizes[i+1] Neuronen, jedes mit layer_sizes[i] Inputs
        self.layers = [Layer(self.layer_sizes[i + 1], self.layer_sizes[i])
                       for i in range(len(self.layer_sizes) - 1)]

    # ---------------------------------------------------------------- Rechnen
    def predict(self, inputs):
        """Nur Forward-Pass: Output jeder Schicht wird Input der nächsten."""
        values = numpy.asarray(inputs, dtype=float)
        for layer in self.layers:
            values = layer.forward(values)
        return values

    def train(self, inputs, targets):
        """Ein SGD-Schritt mit EINEM Sample. Gibt die Kosten J zurück."""
        learning_rate = self.learning_rate
        # 1) Forward-Pass (die Neuronen merken sich Inputs/Outputs)
        output = self.predict(inputs)
        targets = numpy.asarray(targets, dtype=float)
        # 2) Kostenfunktion MSE: J = 1/2 * Summe (y_hat - y)^2
        cost = 0.5 * numpy.sum((output - targets) ** 2)
        # 3) Ableitung dJ/dy_hat = y_hat - y, dann rückwärts durch alle Layer
        errors = output - targets
        for layer in reversed(self.layers):
            errors = layer.backward(errors, learning_rate)
        return cost

    def fit(self, samples, targets, epochs, shuffle=True, log_every=0):
        """Trainingsschleife: pro Epoche jedes Sample einmal (Stochastic GD)."""
        order = numpy.arange(len(samples))
        for epoch in range(1, epochs + 1):
            if shuffle:
                numpy.random.shuffle(order)
            cost = sum(self.train(samples[i], targets[i])
                       for i in order) / len(samples)
            if log_every and epoch % log_every == 0:
                print(f"Epoche {epoch:6d}  Kosten {cost:.6f}")
        return cost

    # ---------------------------------------------------------------- Export
    def to_dict(self):
        return {
            "layer_sizes": self.layer_sizes,
            "layers": [
                [{"weights": n.weights.tolist(), "bias": float(n.bias)}
                 for n in layer.neurons]
                for layer in self.layers
            ],
        }

    def dump_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data):
        net = cls(data["layer_sizes"])
        for layer, layer_data in zip(net.layers, data["layers"]):
            for neuron, n_data in zip(layer.neurons, layer_data):
                neuron.weights = numpy.array(n_data["weights"], dtype=float)
                neuron.bias = float(n_data["bias"])
        return net

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def dump_txt(self, path):
        """Lesbares Textformat (auch für die C-Version in Part 4 gedacht).

        Aufbau:  Anzahl Layer-Größen, die Größen, dann pro Neuron eine Zeile
                 'Bias Gewicht1 Gewicht2 ...'
        """
        lines = [f"layers {len(self.layer_sizes)}",
                 "sizes " + " ".join(str(s) for s in self.layer_sizes)]
        for i, layer in enumerate(self.layers, start=1):
            lines.append(f"layer {i} neurons {len(layer.neurons)} inputs {self.layer_sizes[i - 1]}")
            for j, n in enumerate(layer.neurons, start=1):
                weights = " ".join(f"{w:.10f}" for w in n.weights)
                lines.append(f"neuron {j} bias {n.bias:.10f} weights {weights}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def dump_initfile(self, path, name="net"):
        """C-Header mit allen Konstanten, damit ein C-Programm das Netz initialisieren kann."""
        guard = name.upper() + "_H"
        out = [f"#ifndef {guard}", f"#define {guard}", "",
               f"#define {name.upper()}_NUM_LAYERS {len(self.layers)}",
               f"const int {name}_input_size = {self.layer_sizes[0]};",
               f"const int {name}_layer_sizes[{len(self.layers)}] = {{"
               + ", ".join(str(s) for s in self.layer_sizes[1:]) + "};", ""]
        for i, layer in enumerate(self.layers):
            n_out, n_in = len(layer.neurons), self.layer_sizes[i]
            biases = ", ".join(f"{n.bias:.10f}" for n in layer.neurons)
            out.append(f"const double {name}_l{i}_bias[{n_out}] = {{{biases}}};")
            rows = ",\n    ".join("{" + ", ".join(f"{w:.10f}" for w in n.weights) + "}"
                                  for n in layer.neurons)
            out.append(f"const double {name}_l{i}_weights[{n_out}][{n_in}] = {{\n    {rows}\n}};")
            out.append("")
        out.append(f"#endif /* {guard} */")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
