import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras import models, layers, losses, optimizers, metrics, activations

tf.config.experimental.set_visible_devices([], 'GPU') # Fixed tf actual GPU bug, be carefull it disable GPU

vino = load_wine()
print(vino['DESCR'])

caracteristicas = vino['data']
objetivo = vino['target']

# Train Model #
x_train, x_test, y_train, y_test = train_test_split(caracteristicas, objetivo, test_size=0.3)

normalizador = MinMaxScaler()
x_train_normalizado = normalizador.fit_transform(x_train)
x_test_normalizado =normalizador.transform(x_test)

modelo = models.Sequential()
modelo.add(layers.Dense(units=13, input_dim=13, activation='relu'))
modelo.add(layers.Dense(units=13, activation='relu'))
modelo.add(layers.Dense(units=13, activation='relu'))
modelo.add(layers.Dense(units=3, activation='softmax')) # 3 clases de vino

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Test model #
modelo.fit(x_train_normalizado, y_train, epochs=60)

predicciones = modelo.predict_classes(x_test_normalizado)
print(classification_report(y_test, predicciones))