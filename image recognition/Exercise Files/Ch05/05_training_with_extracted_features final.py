from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import joblib
import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU') # Fixed tf actual GPU bug, be carefull it disable GPU

# Load data set
x_train = joblib.load("Ch05/x_train.dat")
y_train = joblib.load("Ch05/y_train.dat")

# Create a model and add layers
model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=10,
    shuffle=True
)

# Save neural network structure
model_structure = model.to_json()
f = Path("Ch05/model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("Ch05/model_weights.h5")
