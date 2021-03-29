#Aula 3.2 - K-Means
#Clusterização com K-Means
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Leitura dos dados
df = pd.read_csv('Mall_Customers.csv')
df.head()
df.shape

#Verificar dados nulos
df.isnull().sum()

#Informações estatísticas
df.describe()

#Gerando gráfico com a renda anual versus o score dos clientes.
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], marker = '*'), plt.xlabel('Renda anual (k$)'), plt.ylabel('Score (1-100)'), plt.show()

#Selecionando dados para o agrupamento
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
X.head()

#Importando K-Means
from sklearn.cluster import KMeans

#Clusterizando com k = 5
modelo_kmeans = KMeans(n_clusters = 5, init = 'k-means++')
y_kmeans = modelo_kmeans.fit_predict(X)

#Visualizando o primeiro grupo criado
X[y_kmeans == 0]

#Visualizando os grupos
k_grupos = 5
cores = ['r', 'b', 'k', 'y', 'g']
for k in range(k_grupos): 
    cluster = X[y_kmeans == k]
    plt.scatter(cluster['Annual Income (k$)'], cluster['Spending Score (1-100)'], s = 100, c = cores[k], label = f'Cluster {k}')
    
plt.title('Grupos de clientes'), plt.xlabel('Renda anual (k$)'), plt.ylabel('Score (1-100)'), plt.grid(), plt.legend(), plt.show()

plt.scatter()