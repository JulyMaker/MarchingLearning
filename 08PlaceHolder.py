import tensorflow as tf

#tensor = tf.compat.v1.random_uniform((5,5),0,1)
#variable = tf.Variable(initial_value=tensor)
#
#inicializador = tf.compat.v1.global_variables_initializer()
#
#with tf.compat.v1.Session() as sesion:
#    sesion.run(inicializador)
#    resultado = sesion.run(variable)
#
#print(resultado)
tf.compat.v1.disable_eager_execution()

incognitas = tf.compat.v1.placeholder(tf.float32, shape=(20,20))

print(incognitas)
