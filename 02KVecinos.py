import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#TrainModel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#TestModel
from sklearn.metrics import classification_report, confusion_matrix

vehiculos = pd.read_csv('./resources/vehiculos.csv')

#vehiculos['vehicle_class'].unique()

x = vehiculos.drop('vehicle_class', axis=1)
y = vehiculos['vehicle_class']

# Train Model #
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)
predicciones = knn.predict(x_test)

#print(predicciones)

print(classification_report(y_test, predicciones))

print(confusion_matrix(y_test, predicciones))

tasa_error = []
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    predicciones_i = knn.predict(x_test)
    tasa_error.append(np.mean(predicciones_i != y_test))

valores = range(1,30)
plt.plot(valores, tasa_error, 'g', marker='o', markerfacecolor='red', markersize=8)
plt.show()