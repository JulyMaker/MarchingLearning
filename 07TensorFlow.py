import tensorflow as tf

tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x
# Build a graph.
mensaje1 = tf.constant("Hola ")
mensaje2 = tf.constant(" Mundo")

# Launch the graph in a session.
with tf.compat.v1.Session() as sesion:
    resultado = sesion.run(mensaje1 + mensaje2).decode() #It is a bytestring

print(resultado)


# OPERATIONS #
#constante = tf.constant(15)
#matrix1 = tf.fill((6,6),10)
#matrix2 = tf.random_normal((5,5))
#matrix3 = tf.random_uniform((4,4), minval=0, maxval=5)
#matrix_ceros = tf.zeros((2,2))
#matrix_unos = tf.ones((3,3))

#operaciones = [constante, matrix1, matrix2, matrix3, matrix_ceros, matrix_unos]

#with tf.compat.v1.Session() as sesion:
#    for op in operaciones:
#        print(sesion.run(op))
#        print("\n")


# GRAFOS #

#nodo1 = tf.constant(4)
#nodo2 = tf.constant(6)
#nodo3 = nodo1 + nodo2
#
#with tf.compat.v1.Session() as sesion:
#    resultado = sesion.run(nodo3)

# DEFAULT GRAPH #

# print(tf.get_default_graph())
# grafo1 = tf.Graph()
#with grafo1.as_default():
#    print(grafo1 is tf.get_default_graph())