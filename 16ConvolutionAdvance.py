import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Convolution2D(128, (3, 3), strides=(1,1), input_shape=(64,64,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64, (3,3), strides=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Definir Datagens
train_data_gen = ImageDataGenerator(
  rescale = 1./255, #Normalizacion para que los valores de las imagenes para que queden entre 0 y 1
  shear_range = 0.2, #Direccion de rotacion contrareloj
  zoom_range = 0.2, #Aplicacion de zoom aleatorio
  horizontal_flip = True # Va a invertir horizontalmente la imagen aleatoriamente
)

test_datagen = ImageDataGenerator(
  rescale = 1./255 #Normalizacion para que los valores de las imagenes para que queden entre 0 y 1
)
#Armar datasets
training_set = train_data_gen.flow_from_directory(
  "train",
  target_size = (64, 64), #Valores de nuestras capas de entrada (64, 64, 3)
  batch_size = 256,
  class_mode = "binary"
)

test_set = test_datagen.flow_from_directory("valid",
                                           target_size = (64,64),
                                           batch_size = 256,
                                           class_mode = "binary"
                                           )


#Entrenar
model.fit_generator(
  training_set,
  epochs = 25,
  validation_data = test_set,
)

#Predecir
#from google.colab.files import upload #Colab
#import numpy as np
##upload()
#from tensorflow.keras.preprocessing import image
#imagen = image.load_img('./resources/p.jpg', target_size=(64,64))
#imagen = image.img_to_array(imagen)
#imagen = np.expand_dims(imagen, axis=0) #(64, 64, 3) -> (1, 64, 64, 3) 
#res = model.predict_classes(imagen)
#print('Resultado: ', res)
#print('Etiquetas: ',training_set.class_indices)
#
##Guardar y cargar modelo
#from google.colab import drive
#drive.mount('/content/gdrive')
#os.mkdir('/content/gdrive/My Drive/Deep Models/Dog Vs Cat')
#model.save('/content/gdrive/My Drive/Deep Models/Dog Vs Cat/DogVSCatModel.h5')
#
#from tensorflow.keras.models import load_model
#model = load_model('/content/gdrive/My Drive/Deep Models/Dog Vs Cat/DogVSCatModel.h5')