import tensorflow as tf
import matplotlib.pyplot as plt
#import tensorflow.examples.tutorials.mnist as input_data

input_data = tf.keras.datasets.mnist
# Load filea #
#mnist = input_data.read_data_sets('.resources/convolucion/', one_host=True)
#mnist.train.num_examples
#mnist.test.num_examples
#imagen = mnist.train.imagen[1] 28x28
#imagen = imagen.reshape(28,28)
#plt.imshow(imagen)

#load mnist data
(x_train, y_train), (x_test, y_test) = input_data.load_data()

def create_mnist_dataset(data, labels, batch_size):
  def gen():
    for image, label in zip(data, labels):
        yield image, label
  ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((28,28 ), ()))

  return ds.repeat().batch(batch_size)

#train and validation dataset with different batch size
train_dataset = create_mnist_dataset(x_train, y_train, 10)
valid_dataset = create_mnist_dataset(x_test, y_test, 20)

print(len(x_train)) # 60.000 images
plt.imshow(x_train[5000], cmap='gray_r') 
plt.show()