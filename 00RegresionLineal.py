import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#www.kaggle.com
casas = pd.read_csv('E:/personal/Repos/MarchingLearning/USA_Housing.csv')

casas.head(10)
#casas.info()
#casas.describe()
#casas.columns
#casas['Price']
#casas['Avg. Area Number of Rooms']

sns.distplot(casas['Price'])
sns.heatmap(casas.corr(), annot=True)
plt.show()