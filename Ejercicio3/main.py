import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir la función f(x1, x2)
def f(x1, x2):
    return 10 - np.exp(-(x1**2 + 3*x2**2))

# Gradiente de la función f(x1, x2)
def gradient_f(x1, x2):
    df_dx1 = 2*x1 * np.exp(-(x1**2 + 3*x2**2))
    df_dx2 = 6*x2 * np.exp(-(x1**2 + 3*x2**2))
    return np.array([df_dx1, df_dx2])

# Función para el descenso del gradiente
def gradient_descent(lr, max_iters):
    # Inicializar valores aleatorios para x1 y x2 dentro de los límites [-1, 1]
    x = np.random.uniform(-1, 1, size=2)
    
    # Almacenar los puntos para graficar el proceso de optimización
    x_history = [x.copy()]
    
    # Iterar hasta convergencia o alcanzar el número máximo de iteraciones
    for _ in range(max_iters):
        # Calcular el gradiente de la función en el punto actual
        grad = gradient_f(x[0], x[1])
        
        # Actualizar los valores de x utilizando el descenso de gradiente
        x -= lr * grad
        
        # Aplicar los límites [-1, 1]
        x = np.clip(x, -1, 1)
        
        # Guardar el punto actual
        x_history.append(x.copy())
        
    return x, np.array(x_history)

# Configuración de hiperparámetros
learning_rate = 0.1
max_iterations = 1000

# Ejecutar el descenso del gradiente
optimal_point, history = gradient_descent(learning_rate, max_iterations)

# Imprimir el punto óptimo encontrado
print("Punto optimo encontrado:")
print("x1:", optimal_point[0])
print("x2:", optimal_point[1])
print("Valor optimo de f(x1, x2):", f(*optimal_point))

# Visualizar la función y el proceso de optimización
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
x1, x2 = np.meshgrid(x1, x2)
z = f(x1, x2)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, z, cmap='viridis', alpha=0.8)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Función de optimización')

# Mostrar los puntos del proceso de optimización
ax.plot(history[:, 0], history[:, 1], f(history[:, 0], history[:, 1]), marker='o', color='r', linestyle='-')
ax.scatter(optimal_point[0], optimal_point[1], f(*optimal_point), color='g', s=100, label='Punto óptimo')

plt.legend()
plt.show()

