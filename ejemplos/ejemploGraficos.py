import numpy as np
import pandas as pd

dataframe = pd.DataFrame(np.random.randint(200, size=(50,4)), columns=['a','b','c','d'])
dataframe

dataframe['a'].hist()
dataframe['b'].hist()
dataframe['b'].hist(bins=30)

dataframe['a'].plot.hist()

dataframe.plot.area(alpha=0.3)

dataframe.plot.bar()

dataframe.plot.bar(stacked=True)

dataframe.plot.scatter(x='a', y='b', c='c', cmap='coolwarm')

dataframe.plot.box()

dataframe.plot.hexbin(x='a', y='b', gridsize=12)

dataframe.plot.kde()
dataframe.plot.density()
