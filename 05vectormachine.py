import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#TrainModel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 

#TestModel
from sklearn.metrics import classification_report, confusion_matrix

flores = sns.load_dataset('iris')

#sns.pairplot(flores)
#sns.pairplot(flores, hue='species')
print(flores.columns)

x = flores.drop('species', axis=1)
y = flores['species']

# Train Model #
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

modelo = SVC(gamma='auto')
modelo.fit(x_train, y_train)

# Test Model #
predicciones = modelo.predict(x_test)

print(classification_report(y_test, predicciones))
print(confusion_matrix(y_test, predicciones))