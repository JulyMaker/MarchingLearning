import pandas as pd
import numpy as np
#import cufflinks as cf
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import matplotlib.pyplot as plt
import plotly.express as px

#init_notebook_mode(connected=True)
#cf.go_offline()
#%matplotlib inline

dataframe = pd.DataFrame(np.random.randn(100,4), columns=['a','b','c','d'])
#dataframe.plot()
#plt.show()

#dataframe.iplot()
#dataframe.iplot(kind='box')
#dataframe.iplot(kind='surface')
#dataframe['a'].iplot(kind='hist', bins=30)
fig = px.histogram(dataframe, x="a", nbins=30)
fig.show()