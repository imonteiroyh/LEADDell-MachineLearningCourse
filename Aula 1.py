#Aula 1.1 - Python aplicado a Data Science
#Introdução à análise de dados com Python
import pandas as pd
frutas = pd.read_table('fruit_data_with_colors.txt', sep = '\t')
frutas.shape
frutas.head(5)

#Análise estatística de um conjunto de dados
frutas.describe()['mass']
frutas.describe()['mass']['min']
frutas.describe()[['mass', 'color_score']]
frutas['mass']
frutas[['mass', 'color_score']]
frutas[10:15]
frutas[['mass', 'color_score']][10:15]
frutas[['mass', 'color_score']][10:15].describe()

#Análise e visualização de um conjunto de dados
import matplotlib.pyplot as plt
freq = frutas['fruit_name'].value_counts()
freq.plot(kind = 'bar'), plt.show()
frutas['fruit_name'] == 'apple'
macas = frutas['fruit_name'] == 'apple'
frutas[macas]
pesadas = frutas['mass'] > 175
frutas[macas & pesadas]
X1 = frutas[macas & pesadas]['width']
X2 = frutas[macas & pesadas]['height']
plt.scatter(X1, X2), plt.xlabel('Comprimento'), plt.ylabel('Altura'), plt.show()
X1 = frutas['width']
X2 = frutas['height']
plt.scatter(X1, X2), plt.xlabel('Comprimento'), plt.ylabel('Altura'), plt.show()
Y = frutas['fruit_label']
plt.scatter(X1, X2, c = Y), plt.xlabel('Comprimento'), plt.ylabel('Altura'), plt.show()

#Aula 1.3 - Pré-processamento de dados
#Introdução ao pré-processamento de dados com Python e imputação de valores faltantes
data = pd.read_table('fruit_data_with_colors_miss.txt', na_values = ['.', '?'])
data = data.fillna(data.mean())
data['fruit_subtype'].fillna(data['fruit_subtype'].value_counts().idxmax(), inplace = True)

#Eliminação de colunas
data.isnull().sum()
data.isnull().sum()/data.shape[0]*100
data = data[['mass', 'width', 'height', 'color_score']]

#Transformando a escala dos dados
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
data_escala = mm.fit_transform(data)

#Encontrando outliers
data = pd.read_table('fruit_data_with_colors_miss.txt', na_values = ['.', '?'])
data = data.fillna(data.mean())
macas = data[data['fruit_name'] == 'apple'] 
est = macas['mass'].describe()
macas[(macas['mass'] > est['mean'] + est['std']*2)]
macas[(macas['mass'] < est['mean'] - est['std']*2)]