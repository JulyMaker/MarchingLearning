import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#TrainModel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#TestModel
from sklearn.metrics import classification_report, confusion_matrix

vinos = pd.read_csv('./resources/vinos.csv')

print(vinos.columns)

x = vinos.drop('Wine Type', axis=1)

y= vinos['Wine Type']

# Train Model #
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)

arbol = DecisionTreeClassifier()
arbol.fit(x_train, y_train)

predicciones = arbol.predict(x_test)

print(classification_report(y_test, predicciones))
print(confusion_matrix(y_test, predicciones))