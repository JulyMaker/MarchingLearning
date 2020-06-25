import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#TrainModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#TestModel
from sklearn import metrics

#www.kaggle.com
casas = pd.read_csv('E:/personal/Repos/MarchingLearning/USA_Housing.csv')

casas.head()
#casas.info()
#casas.describe()
#casas.columns
#casas['Price']
#casas['Avg. Area Number of Rooms']

#sns.distplot(casas['Price'])
#sns.heatmap(casas.corr(), annot=True)
#plt.show()

print(casas.columns)

x = casas[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = casas['Price']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

# Train model

lrm = LinearRegression()
lrm.fit(x_train, y_train)

# Test model

predicciones = lrm.predict(x_test)

plt.scatter(y_test, predicciones)   #Diagonal good model
plt.show()

sns.distplot(y_test - predicciones) #Error - Gaus bell good model
plt.show()

# MAE (Mean asolute error) - Media del valor absoluto de los errores
mae= metrics.mean_absolute_error(y_test, predicciones) # Minor is better
print("MAE = {}".format(mae))

#MSE (Media de los errores al cuadrado)
mse = metrics.mean_squared_error(y_test, predicciones)
print("MSE = {}".format(mse))

#RMSE (Raiz cuadrada de la media de los errores al cuadrado)
rmse = np.sqrt(mse)
print("RMSE = {}".format(rmse))