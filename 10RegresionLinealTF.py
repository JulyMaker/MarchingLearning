import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.config.experimental.set_visible_devices([], 'GPU') # Fixed tf actual GPU bug, be carefull it disable GPU

datos_x = np.linspace(0,10,10) + np.random.uniform(-1,1,10)
datos_y = np.linspace(0,10,10) + np.random.uniform(-1,1,10)

# y = mx + b #

print(np.random.rand(2))
m = tf.Variable(0.81) # random1
b = tf.Variable(0.87) # random2

print("Valor de m: ")
print(m)
print("Valor de b: ")
print(b)

def errorfnc():
    error = 0
    for x,y in zip(datos_x, datos_y):
        y_pred = m*x + b
        error = error + (y - y_pred)**2
    return error

#optimizador = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
optimizador = tf.keras.optimizers.SGD(learning_rate=0.001)

#Best m and b values, it minimizes the error# 
pasos = 3
for i in range(pasos):
    optimizador.minimize(errorfnc, var_list=[m, b])

final_m, final_b = [m,b]

x_test = np.linspace(-1,11,10)
y_pred_2 = (final_m * x_test) + final_b

print("Valor de m: ")
print(m)
print("Valor de b: ")
print(b)

plt.plot(x_test, y_pred_2, 'r')
plt.plot(datos_x, datos_y, '*')
plt.show()