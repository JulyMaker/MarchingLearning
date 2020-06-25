import pandas as pd
from sqlalchemy import create_engine

diccionario = {'A':[1,2,3], 'B':[4,5,6]}
dataframe = pd.DataFrame(diccionario)
dataframe

engine = create_engine('sqlite:///:memory:')
dataframe.to_sql('tabla',engine,index=False)

datos_leidos_bd = pd.read_sql('tabla',con=engine)
datos_leidos_bd