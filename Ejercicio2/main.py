import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

class PerceptronSimple:
    def __init__(self, num_caracteristicas, tasa_aprendizaje=0.01, epocas=100):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.pesos = np.zeros(num_caracteristicas + 1)  # +1 por el sesgo

    def predecir(self, x):
        activacion = np.dot(self.pesos[1:], x) + self.pesos[0]
        return 1 if activacion >= 0 else -1

    def entrenar(self, X, y):
        for _ in range(self.epocas):
            for i in range(X.shape[0]):
                prediccion = self.predecir(X[i])
                self.pesos[1:] += self.tasa_aprendizaje * (y[i] - prediccion) * X[i]
                self.pesos[0] += self.tasa_aprendizaje * (y[i] - prediccion)

    def evaluar(self, X_test, y_test):
        correcto = 0
        for i in range(X_test.shape[0]):
            if self.predecir(X_test[i]) == y_test[i]:
                correcto += 1
        return correcto / X_test.shape[0]

# Función para cargar datos desde un archivo CSV
def cargar_csv(filename):
    data = pd.read_csv(filename, header=None)
    return data.values

# Cargar datos originales
datos_originales = cargar_csv('spheres1d10.csv')

# Función para perturbar los datos con un cierto porcentaje
def perturbar_datos(data, porcentaje):
    ruido = np.random.normal(0, 0.1, size=data.shape)
    return data + porcentaje * ruido

# Cargar datos perturbados con diferentes porcentajes
perturbado_10 = perturbar_datos(cargar_csv('spheres2d10.csv'), 0.1)
perturbado_50 = perturbar_datos(cargar_csv('spheres2d50.csv'), 0.5)
perturbado_70 = perturbar_datos(cargar_csv('spheres2d70.csv'), 0.7)

# Gráficas en 3D
fig = plt.figure(figsize=(15, 15))

# Datos originales
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(datos_originales[:,0], datos_originales[:,1], datos_originales[:,2], c='m')
ax1.set_title('Datos originales')

# Datos perturbados < 10%
ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(perturbado_10[:,0], perturbado_10[:,1], perturbado_10[:,2], c='m')
ax2.set_title('Datos perturbados < 10%')

# Datos perturbados < 50%
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(perturbado_50[:,0], perturbado_50[:,1], perturbado_50[:,2], c='m')
ax3.set_title('Datos perturbados < 50%')

# Datos perturbados < 70%
ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(perturbado_70[:,0], perturbado_70[:,1], perturbado_70[:,2], c='m')
ax4.set_title('Datos perturbados < 70%')

plt.show()

