import seaborn as sns
import matplotlib.pyplot as plt

propinas = sns.load_dataset('tips')
propinas.head(10)

sns.distplot(propinas['total_bill'], bins=20)

sns.jointplot(x='total_bill', y='tip', data=propinas, kind='hex')
sns.jointplot(x='total_bill', y='tip', data=propinas, kind='reg')
sns.jointplot(x='total_bill', y='tip', data=propinas, kind='kde')

sns.pairplot(propinas)
sns.pairplot(propinas, hue='sex')

sns.rugplot(propinas['total_bill'])

sns.barplot(x='sex', y='total_bill', data=propinas)
sns.countplot(x='sex', data=propinas)
sns.boxplot(x='day', y='total_bill', data=propinas, hue='smoker')
sns.violinplot(x='day', y='total_bill', data=propinas, hue='smoker', split=True)
#stripplot
#svarmplot
plt.show()

#Mapas de calor
vuelos = sns.load_dataset('flights')
vuelos.head(10)

vuelos_matrix = vuelos.pivot_table(index='month', columns='year',values='passengers')
print(vuelos_matrix)

sns.heatmap(vuelos_matrix)
#sns.heatmap(vuelos_matrix, cmap='viridis')
#viridis, plasma, inferno, magma, coolwarm
#sns.heatmap(vuelos_matrix, cmap='viridis', linecolor='white', linewidths=2)
plt.show()