#Aula 2.1 - Introdução à Regressão
#Importação de bibliotecas e registro dos dados
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model._glm.glm import _y_pred_deviance_derivative
temperatura = np.array([30, 25, 36, 18, 25, 29, 30, 33, 37, 31, 26, 37, 29, 26, 30, 31, 34, 38])
numero_sorvetes = np.array([20, 12, 50, 10, 18, 25, 26, 32, 48, 22, 16, 52, 24, 20, 28, 29, 35, 40])
df = pd.DataFrame({'temperatura':temperatura, 'numero_sorvetes':numero_sorvetes})
plt.plot(df['temperatura'], df['numero_sorvetes'], '*'), plt.xlabel('Temperatura'), plt.ylabel('Número de Sorvetes'), plt.show()

#Separação de variável independente e variável dependente
x = df['temperatura'].to_numpy()
y = df['numero_sorvetes'].to_numpy()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#Treinando o modelo
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()
modelo.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

#Realizando a previsão do número de sorvetes a serem vendidos
y_pred = modelo.predict(x_test.reshape(-1, 1))
plt.plot(range(y_pred.shape[0]), y_pred, 'r--'), plt.plot(range(y_test.shape[0]), y_test, 'g--'), plt.legend(['Sorvetes previstos', 'Sorvetes vendidos']), plt.xlabel('Índice'), plt.ylabel('Número de Sorvetes'), plt.show() 

#Aula 2.3 - KNN (K-Nearest Neighbors)
#KNN para classificação
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
data = pd.read_table('fruit_data_with_colors.txt')
X = data[['mass', 'width', 'height',  'color_score']]
y = data['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

#Pré-processando os dados para o KNN
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
knn.predict(X_test)

#KNN para regressão
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 3)
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y = True)
print(load_boston()['DESCR'])
X_train, X_test, y_train, y_test = train_test_split(X, y)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

#Aula 2.4 - Decision Trees
#Implementando as árvores de decisão
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_boston, load_iris

#Árvore de decisão para classificação
dt = DecisionTreeClassifier()
X, y = load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
dt.fit(X_train, y_train)
dt.score(X_test, y_test)

#Árvore de decisão para regressão
dt = DecisionTreeRegressor()
X, y = load_boston(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
dt.fit(X_train, y_train)
dt.score(X_test, y_test)

#Evitando o overfitting
dt = DecisionTreeClassifier(max_depth = 3)
X, y = load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
dt.fit(X_train, y_train)
dt.score(X_test, y_test)

#Selecionando features com árvores de decisão
from sklearn.datasets import load_breast_cancer
dt = DecisionTreeClassifier()
X, y = load_breast_cancer(return_X_y = True)
dt.fit(X, y)
dt.feature_importances_

#Aula 2.5 - Regressão Logística
#Leitura dos dados de diabetes
df = pd.read_csv('diabetes.csv')
df.describe()

#Gerando gráfico
plt.plot(df['Age'], df['Pregnancies'], 'o'), plt.xlabel('Idade'), plt.ylabel('Gravidez'), plt.show()

#Separação dos dados
y = df['Outcome']
x = df.drop('Outcome', axis = 1)

#Separando os dados em train e test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#Selecionar modelo de regressão logística
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 5000)

#Treinamento do modelo
model.fit(x_train, y_train)

#Realizando as previsões
y_pred = model.predict(x_test)

#Validando modelo
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy*100