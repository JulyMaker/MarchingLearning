import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px

#TrainModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#TestModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


entrenamiento = pd.read_csv('./resources/train.csv')
pruebas = pd.read_csv('./resources/test.csv')

# Graphics #

#entrenamiento.head()
#sns.heatmap(entrenamiento.isnull())
#sns.countplot(x='Survived', data=entrenamiento)
#sns.countplot(x='Survived', data=entrenamiento, hue='Sex')
#sns.countplot(x='Survived', data=entrenamiento, hue='Pclass')
#sns.distplot(entrenamiento['Age'], dropna(), kde=False, bins=30)
#entrenamiento['Age'].plot.hist(bins=30)
#entrenamiento['SibSp'].plot.hist(bins=30)
#plt.show()

print(entrenamiento.columns)

#fig = entrenamiento['Fare'].iplot(kind='hist', bins=40)
fig = px.histogram(entrenamiento, x="Fare", nbins=40)
#fig = px.box(entrenamiento, y="Fare")
#fig.show()


# Clean dataset #
# sns.heatmap(entrenamiento.isnull())

# Med age
sns.boxplot(x='Pclass', y='Age', data=entrenamiento)

# Function
def edad_media(columnas):
    edad = columnas[0]
    clase = columnas[1]
    if(pd.isnull(edad)):
        if clase == 1:
            return 38
        elif clase == 2:
            return 30
        else:
            return 25
    else:
        return edad

entrenamiento['Age'] = entrenamiento[['Age','Pclass']].apply(edad_media, axis=1)
entrenamiento.drop('Cabin', axis=1, inplace=True)
entrenamiento.drop(['Name','Ticket','PassengerId'], axis=1, inplace=True)
sexo = pd.get_dummies(entrenamiento['Sex'], drop_first=True)
puerto = pd.get_dummies(entrenamiento['Embarked'], drop_first=True)

sns.heatmap(entrenamiento.isnull())
plt.show()

entrenamiento = pd.concat([entrenamiento, sexo], axis=1)
entrenamiento.drop('Sex', axis=1, inplace=True)
entrenamiento = pd.concat([entrenamiento, puerto], axis=1)
entrenamiento.drop('Embarked', axis=1, inplace=True)

# Train Model #

x = entrenamiento.drop('Survived', axis=1)
y = entrenamiento['Survived']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=45)

modelo = LogisticRegression()
modelo.fit(x_train, y_train)

predicciones = modelo.predict(x_test)

print(classification_report(y_test, predicciones))

print(confusion_matrix(y_test, predicciones))