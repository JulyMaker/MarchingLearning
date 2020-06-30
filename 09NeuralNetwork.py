import tensorflow as tf
import numpy as np
import sys

tf.config.experimental.set_visible_devices([], 'GPU') # Fixed tf actual GPU bug, be carefull it disable GPU

#my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
#my_variable = tf.Variable(my_tensor)
#
## Variables can be all kinds of types, just like tensors
#bool_variable = tf.Variable([False, False, False, True])
#complex_variable = tf.Variable([5 + 4j, 6 + 1j])
#
#print("Shape: ",my_variable.shape)
#print("DType: ",my_variable.dtype)
#print("As NumPy: ", my_variable.numpy)

# Uncomment to see where your variables get placed (see below)
# tf.debugging.set_log_device_placement(True)

#tensor = tf.random.uniform((5,5),0,1)
#variable = tf.Variable(tensor)
#
#print("\nViewed as a tensor:", tf.convert_to_tensor(variable))
#print(variable.numpy)

# Whitout placeholders #
#aleatorio_a = np.random.uniform(0,50,(4,4))
#aleatorio_b = np.random.uniform(0,50,(4,1))
#
#suma = aleatorio_a + aleatorio_b
#multiplicacion = aleatorio_a * aleatorio_b
#
#print(suma)
#print(multiplicacion)

# Neural Network #

caracteristicas = 10
neuronas = 4

x = np.float32(np.random.random_sample([1,caracteristicas]))
w = tf.Variable(tf.random.normal([caracteristicas, neuronas]))
b = tf.Variable(tf.ones([neuronas]))

multiplicacion =tf.matmul(x,w)
z = tf.add(multiplicacion,b)
activacion = tf.sigmoid(z)

print(activacion)