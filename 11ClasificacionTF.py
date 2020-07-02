import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load csv file #
ingresos = pd.read_csv('./resources/ingresos.csv')

# normalize income 0 or 1 #
ingresos['income'].unique()

def cambio_valor(valor):
    if valor == '<=50K':
        return 0
    else:
        return 1

ingresos['income'] = ingresos['income'].apply(cambio_valor)
# print(ingresos.head())

x = ingresos.drop('income', axis=1)
y = ingresos['income']

# Train Model #
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)

print(ingresos.columns)
### Define continuous list
CONTI_FEATURES  = ['age', 'fnlwgt','capital-gain', 'education-num', 'capital-loss', 'hours-per-week']
### Define the categorical list
CATE_FEATURES = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender",['Female','Male'])
race = tf.feature_column.categorical_column_with_vocabulary_list("race",['Black','White'])
continuous_features = [tf.feature_column.numeric_column(k) for k in CONTI_FEATURES]	
categorical_features = [tf.feature_column.categorical_column_with_hash_bucket(k, hash_bucket_size=1000) for k in CATE_FEATURES]
columnas_categorias = categorical_features + continuous_features + [gender,race]

# input function #
'''
Begin
'''
def input_fn(features, labels, batch_size=256, training=True):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)
'''
End
'''

funcion_entrada = input_fn(features=x_train,labels=y_train, batch_size=100)

#print(dir(tf.estimator.LinearClassifier))
modelo = tf.estimator.LinearClassifier(columnas_categorias)
#modelo.train(input_fn=funcion_entrada,steps=8000)
#
## TestModel #
#
#funcion_predicciones = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test,batch_size=len(x_test),shuffle=False)
#generador_predicciones = modelo.predict(input_fn=funcion_predicciones)
#predicciones = list(generador_predicciones)
#
#predicciones_finales = [prediccion['class_ids'][0] for prediccion in predicciones]
#
## Print results #
#print(predicciones_finales)
#print(classification_report(y_test, predicciones_finales))
#print(confusion_matrix(y_test, predicciones_finales))