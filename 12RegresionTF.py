import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

# Load csv file #
casas = pd.read_csv('./resources/precios_casas.csv')


casas_x = casas.drop('median_house_value', axis=1)
casas_y = casas['median_house_value']

print(casas.columns)

# Train Model #
x_train, x_test, y_train, y_test = train_test_split(casas_x, casas_y, test_size=0.3)

normalizador = MinMaxScaler()
normalizador.fit(x_train)

x_train = pd.DataFrame(data=normalizador.transform(x_train),columns=x_train.columns, index=x_train.index)
x_test = pd.DataFrame(data=normalizador.transform(x_test),columns=x_test.columns, index=x_test.index)

### Define continuous list
CONTI_FEATURES  = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income']

continuous_features = [tf.feature_column.numeric_column(k) for k in CONTI_FEATURES]	

# input function #
'''
Begin
'''
def input_fn(features, labels, batch_size=256, num_epoch=1000, training=True):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(num_epoch).repeat()
    
    return dataset.batch(batch_size)
'''
End
'''
funcion_entrada = input_fn(features=x_train,labels=y_train, batch_size=10)

modelo = tf.estimator.DNNRegressor(hidden_units=[10,10,10], feature_columns=continuous_features) # 3 layers 10 nodes
modelo.train(input_fn=funcion_entrada,steps=8000)

# TestModel #

funcion_predicciones = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test,batch_size=10, num_epochs=1, shuffle=False)
generador_predicciones = modelo.predict(input_fn=funcion_predicciones)
predicciones = list(generador_predicciones)

predicciones_finales = [prediccion['median_house_value'][0] for prediccion in predicciones]

# Print results #
print(predicciones_finales)
print(mean_squared_error(y_test, predicciones_finales)**0.5)

print(classification_report(y_test, predicciones_finales))
print(confusion_matrix(y_test, predicciones_finales))