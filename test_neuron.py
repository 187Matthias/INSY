import numpy
from nn import Neuron, sigmoid


def kosten(neuron, x, ziel):
    # MSE für ein Sample: J = 1/2 * (y_hat - y)^2
    return 0.5 * (neuron.activation(x) - ziel) ** 2


numpy.random.seed(1)
n = Neuron(3)
x = numpy.array([0.5, -1.0, 2.0])
ziel = 1.0

# 1) Ausgabe liegt in (0, 1), Sigmoid(0) = 0.5
assert 0 < n.activation(x) < 1
assert sigmoid(0) == 0.5
print("1) Ausgabe im Bereich (0,1): OK ->", n.output)

# 2) Ableitung a*(1-a) stimmt mit numerischer Ableitung überein
z, h = 0.3, 1e-6
numerisch = (sigmoid(z + h) - sigmoid(z - h)) / (2 * h)
assert abs(numerisch - n.activation_function_d(sigmoid(z))) < 1e-8
print("2) Sigmoid-Ableitung: OK")

# 3) Gradient des Updates == numerischer Gradient von J (Gradient-Check)
w0, b0 = n.weights.copy(), n.bias
y = n.activation(x)
delta = (y - ziel) * n.activation_d()          # dJ/dZ
grad_w = delta * n.inputs                      # dJ/dW
num_grad = numpy.zeros(3)
for i in range(3):
    n.weights = w0.copy(); n.weights[i] += 1e-6; jp = kosten(n, x, ziel)
    n.weights = w0.copy(); n.weights[i] -= 1e-6; jm = kosten(n, x, ziel)
    num_grad[i] = (jp - jm) / 2e-6
assert numpy.allclose(grad_w, num_grad, atol=1e-8)
print("3) Gradient-Check: OK")

# 4) Update senkt den Fehler und ändert Gewichte + Bias
n.weights, n.bias = w0.copy(), b0
j_vorher = kosten(n, x, ziel)
n.update((n.output - ziel) * n.activation_d(), 0.1)
j_nachher = kosten(n, x, ziel)
assert j_nachher < j_vorher and not numpy.allclose(n.weights, w0) and n.bias != b0
print(f"4) Update senkt Fehler: {j_vorher:.6f} -> {j_nachher:.6f}: OK")

print("Alle Tests bestanden.")
