#Aula 6.1 - SVM (Support Vector Machines)
#Implementando o SVM de classificação
from numpy.core.numeric import cross
from numpy.lib.npyio import load
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
svm = SVC(kernel = 'linear', C = 1.0)
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

#Implementando o SVM com kerneis
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
svm = SVC(kernel = 'linear')
svm.fit(X_train, y_train)
svm.score(X_test, y_test)
svm_kernel = SVC(kernel = 'poly', degree = 3)
svm_kernel.fit(X_train, y_train)
svm_kernel.score(X_test, y_test)
svm_rbf = SVC(kernel = 'rbf')
svm_rbf.fit(X_train, y_train)
svm_rbf.score(X_test, y_test)

#Fórum da Aula 6.1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
wine_quality_data_frame = pd.read_csv('winequality-red.csv')
wine_quality_data_frame.head()
wine_quality_data_frame.isnull().sum()
wine_quality_data_frame['quality'].replace({3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 1, 8 : 1}, inplace = True)
X = wine_quality_data_frame.drop('quality', axis = 1)
y = wine_quality_data_frame['quality']
mm = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = mm.fit_transform(X_train)
X_test = mm.fit_transform(X_test)
svm_linear_c1 = SVC(kernel = 'linear', C = 1.0)
svm_linear_c1.fit(X_train, y_train)
svm_linear_c1.score(X_test, y_test)

#Aula 6.2 - Avaliação de Modelos de Classificação e Regressão
#Modelos Dummy
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.datasets import load_iris, load_boston
X, y = load_iris(return_X_y = True)
dc = DummyClassifier(strategy = 'stratified')
dc.fit(X, y)
dc.score(X, y)
X, y = load_boston(return_X_y = True)
dr = DummyRegressor(strategy = 'mean')
dr.fit(X, y)
dr.score(X, y)

#Matriz de confusão, recall e precisão
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, classification_report
X, y = load_breast_cancer(return_X_y = True)
dc = DummyClassifier(strategy = 'stratified')
dc.fit(X, y)
confusion_matrix(y, dc.predict(X))
print(classification_report(y, dc.predict(X)))

#Validação cruzada
from sklearn.model_selection import cross_val_score
X, y = load_iris(return_X_y = True)
dc = DummyClassifier(strategy = 'stratified')
cross_val_score(dc, X, y, cv = 5)
cross_val_score(dc, X, y, cv = 5).mean()

#Aula 6.3 - Random Forest e Gradient Boosted Decision Trees
#Introdução ao Random Forest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
X, y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
rf = RandomForestRegressor(n_estimators = 100)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

#Introdução ao Gradient Boosted Decision Trees
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
X, y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
gb = GradientBoostingClassifier(n_estimators = 100, max_depth = 3)
gb.fit(X_train, y_train)
gb.score(X_test, y_test)
X, y = load_boston(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
gb = GradientBoostingRegressor(n_estimators = 100, max_depth = 3)
gb.fit(X_train, y_train)
gb.score(X_test, y_test)

#Aula 6.4 - PCA (Principal Component Analysis)
#Implementação do PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y = True)
X.shape
pca = PCA(n_components = 10)
X_pca = pca.fit_transform(X)
X_pca.shape
plt.plot(X_pca[:, 0], X_pca[:, 1]), plt.show()
plt.scatter(X_pca[:, 0], X_pca[:, 1]), plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_pca, y)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
