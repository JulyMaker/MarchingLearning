# No supervised #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#TrainModel
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#TestModel
from sklearn.metrics import classification_report, confusion_matrix

datos = make_blobs(n_samples=200, n_features=2, centers=4)

#plt.scatter(datos[0][:,0], datos[0][:,1])

# Train Model #
modelo = KMeans(n_clusters=4)
modelo.fit(datos[0])

# Test Model #
modelo.cluster_centers_
modelo.labels_

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))

ax1.scatter(datos[0][:,0], datos[0][:,1], c=modelo.labels_)
ax1.set_title('Algoritmo de K-medias')

ax2.scatter(datos[0][:,0], datos[0][:,1], c=datos[1])
ax2.set_title('Datos originales')
plt.show()