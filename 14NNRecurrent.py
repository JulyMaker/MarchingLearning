import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler


tf.config.experimental.set_visible_devices([], 'GPU') # Fixed tf actual GPU bug, be carefull it disable GPU

# RNN con capa de 3 neuronas dasarrollado 2 veces #
numero_entradas = 2
numero_neuronas = 3

x0 = np.float32(np.array([[0,1],[2,3],[4,5]]))
x1 = np.float32(np.array([[2,4],[3,9],[4,1]]))

wx = tf.Variable(tf.Variable(tf.random.normal(shape=[numero_entradas, numero_neuronas])))
wy = tf.Variable(tf.Variable(tf.random.normal(shape=[numero_neuronas, numero_neuronas])))
b  = tf.Variable(tf.zeros([1,numero_neuronas]))

y0 = tf.tanh(tf.matmul(x0,wx) + b)
y1 = tf.tanh(tf.matmul(y0,wy) + tf.matmul(x1,wx) + b)

print(y0)
print(y1)

# EJEMPLO Series temporales #
#tf.compat.v1.disable_eager_execution()
# Load csv file #
leche = pd.read_csv('./resources/produccion_leche.csv', index_col='Month')
leche.index = pd.to_datetime(leche.index)
leche.plot()
plt.show()

conjunto_entrenamiento = leche.head(150)
conjunto_pruebas = leche.tail(18)

normalizador = MinMaxScaler()
entrenamiento_normalizado = normalizador.fit_transform(conjunto_entrenamiento)
pruebas_normalizado = normalizador.transform(conjunto_pruebas)

def lotes(datos_entrenamiento, tamano_lote, pasos):
    comienzo = np.random.randint(0, len(datos_entrenamiento) - pasos )
    lote_y = np.array(datos_entrenamiento[comienzo+pasos+1]).reshape(1, pasos+1)
    return lote_y[:,:-1].reshape(-1,pasos,1), lote_y[:,1:].reshape(-1,pasos,1)

numero_entradass = 1
numero_pasos     = 10
numero_neuronass = 120
numero_salidas   = 1
tasa_aprendizaje = 0.001
tamano_lote      = 1
input_steps      = 5000

decoder_vocab_size = 10
decoder_embedding_dim = 8
encoder_hidden_dim = 6
encoder_output = tf.keras.Input(shape=[None,encoder_hidden_dim],batch_size=tamano_lote) # Input
encoder_seq_length = tf.keras.Input(shape=[], batch_size=tamano_lote, dtype=tf.int32) # Input

#x = tf.placeholder(tf.float32,[None, numero_pasos,numero_entradas])
#y = tf.placeholder(tf.float32,[None, numero_pasos,numero_entradas])

#capa = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=numero_neuronas, activation=tf.nn.relu), output_size=numero_salidas)
#salidas = tf.nn.dynamic_rnn(capa, x, dtype=tf.float32)
#funcion_error = tf.reduce_mean(tf.square(salidas-y))
#optimizador = tf.train.AdamOptimazer(learning_rate=tasa_)
#entrenamiento = optimizador.minimize(funcion_error)

#iint = tf.global_Variables_initializer()
#saver = tf.train.Saver()

#with tf.Session() as sesion:
#   sesion.run(init)
#   for iteraciones in range(input_steps):
#       lote_x, lote_y = lotes(entrenamiento_normalizado, tamano_lote, numero_pasos)
#       sesion.run(entrenamiento, feed_dict={X:lote_x, y:lote_y})
#       if iteracion %100 == 0:
#           error = funcion_error.eval(feed_dict={x:lote_x,y:lote_y})
#           print(iteracion,"\t Error ", error)
#
#       saver.save(sesion, "./modelo_series_temporales")

#with tf.Session() as sesion:
#   saver.restore(sesion, "./modelo_series_temporales")
#   entrenamiento_seed = list(entrenamiento_normalizado[-18:])
#   for iteracion in range(18):
#       lote_x = np.array(entrenamiento_seed[-numero_pasos:]).reshape(1, numero_pasos,1)
#       predicciones_y = sesion.run(salidas, feed_dict={x:lote_x})
#       entrenamiento_seed.append(prediccion_y[0,-1,0])
#  
#resultados = normalizador.inverse_transform(np.array(entrenamiento_seed[18:]).reshape(18,1))
#
#conjunto_pruebas['Predicciones'] = resultados
#conjunto_pruebas.plot()
#plt.show()

lstm_cell = layers.LSTMCell(numero_neuronass) #decoder_cell
attention_mechanism = tfa.seq2seq.BahdanauAttention(numero_neuronass, encoder_output,memory_sequence_length=None)
decoder_init_state = tuple(lstm_cell.get_initial_state(inputs=None, batch_size=tamano_lote, dtype=tf.float32))

attn_cell     = tfa.seq2seq.AttentionWrapper(lstm_cell, attention_mechanism, attention_layer_size=numero_neuronass, initial_cell_state=decoder_init_state, output_attention=False, alignment_history=False) 
output_layer  = layers.Dense(numero_salidas)

att_initial_state = attn_cell.get_initial_state(inputs=None, batch_size=tamano_lote, dtype=tf.float32)
att_initial_state = tfa.seq2seq.AttentionWrapperState(list(att_initial_state.cell_state),att_initial_state.attention,att_initial_state.alignments,
                                                          att_initial_state.alignment_history,att_initial_state.attention_state)

sampler       = tfa.seq2seq.sampler.TrainingSampler()
basic_decoder = tfa.seq2seq.BasicDecoder(cell=attn_cell, sampler=sampler, output_layer=output_layer, output_time_major=True, impute_finished=True, maximum_iterations=input_steps)

decoder_inputs     = tf.keras.Input(shape=[None],batch_size=tamano_lote, dtype=tf.int32)  # Input
decoder_seq_length = tf.keras.Input(shape=[],batch_size=tamano_lote, dtype=tf.int32)  # Input
decoder_embedding  = tf.keras.layers.Embedding(decoder_vocab_size, decoder_embedding_dim, trainable=True) 
decoder_outputs_sequence   = decoder_embedding(decoder_inputs)

final_outputs, last_state, last_sequence_lengths = basic_decoder(decoder_outputs_sequence, initial_state=att_initial_state, sequence_length=decoder_seq_length, training=True)

model = tf.keras.Model([decoder_inputs, decoder_seq_length, encoder_output, encoder_seq_length],[final_outputs, last_state, last_sequence_lengths])

### Test
encoder_timestep = 10
decoder_timestep = 12

encoder_output_data = tf.random.normal(shape=(tamano_lote, encoder_timestep, encoder_hidden_dim))
encoder_seq_length_data = tf.convert_to_tensor([encoder_timestep]*tamano_lote,dtype=tf.int32)


decoder_inputs_data = tf.random.uniform([tamano_lote,decoder_timestep], 0,decoder_vocab_size,tf.int32)
decoder_seq_length_data = tf.convert_to_tensor([decoder_timestep]*tamano_lote,dtype=tf.int32)


a,b,c = model([decoder_inputs_data,decoder_seq_length_data,encoder_output_data,encoder_seq_length_data])  # errors !!!
