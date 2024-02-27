import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def train(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(X[i])
                e = d[i] - y
                self.W = self.W + self.lr * e * x

def read_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',')
    X = data[:, :-1]
    d = data[:, -1]
    return X, d

def plot_points_and_line(X, d, perceptron, title):
    plt.figure()
    plt.title(title)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')

    for i in range(len(d)):
        if d[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], color='blue')
        else:
            plt.scatter(X[i, 0], X[i, 1], color='red')

    slope = -perceptron.W[1] / perceptron.W[2]
    intercept = -perceptron.W[0] / perceptron.W[2]
    x_vals = np.linspace(0, 1, 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, '-g')
    plt.show()

# Lectura de datos de entrenamiento y prueba para XOR
X_train_xor, d_train_xor = read_data("XOR_trn.csv")
X_test_xor, d_test_xor = read_data("XOR_tst.csv")

# Crear y entrenar el perceptrón para XOR
perceptron_xor = Perceptron(input_size=X_train_xor.shape[1], epochs=1000, lr=0.1)
perceptron_xor.train(X_train_xor, d_train_xor)

# Visualización de la separación para XOR en los datos de entrenamiento
plot_points_and_line(X_train_xor, d_train_xor, perceptron_xor, 'Perceptron - XOR Training Data')

# Visualización de la separación para XOR en los datos de prueba
plot_points_and_line(X_test_xor, d_test_xor, perceptron_xor, 'Perceptron - XOR Test Data')

# Datos para la operación OR
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
d_or = np.array([0, 1, 1, 1])

# Crear y entrenar el perceptrón para OR
perceptron_or = Perceptron(input_size=2, epochs=1000, lr=0.1)
perceptron_or.train(X_or, d_or)

# Visualización de la separación para OR
plot_points_and_line(X_or, d_or, perceptron_or, 'Perceptron - OR Data')

