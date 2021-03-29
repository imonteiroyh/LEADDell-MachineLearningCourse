#Aula 5.1 - Introdução às Redes Neurais Artificiais
#Implementando as redes neurais
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston

X, y = load_boston(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
mm = MinMaxScaler()
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
mlp = MLPRegressor(hidden_layer_sizes = (100, 100, 50, 50), max_iter = 1000)
mlp.fit(X_train, y_train)
mlp.score(X_test, y_test)

#Aula 5.2 - Introdução ao Tensorflow
#Preparando o conjunto de dados
import tensorflow as tf
from tensorflow import keras
import numpy as np
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images.shape
len(train_labels)
np.unique(train_labels)
test_images.shape
len(test_labels)
test_labels
train_images = train_images/255.0
test_images = test_images/255.0

#Construção e Avaliação do Modelo de Tensorflow
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])
model.summary()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]
img = test_images[0]
img.shape
img = (np.expand_dims(img, 0))
img.shape
predictions_single = model.predict(img)
predictions_single
np.argmax(predictions_single[0])

#Aula 5.3 - Introdução ao Processamento da Linguagem Natural
#Corpus, tokenização e stopwords
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
frase = "Eu dirijo devagar porque nós queremos ver os animais"
tokens = word_tokenize(frase)
tokens
for t in tokens: 
    if t not in stopwords:
        print(t)

#TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
texto1 = 'A matemática é muito importante para compreendermos como a natureza funciona'
tf_idf = TfidfVectorizer()
vetor1 = tf_idf.fit_transform([texto1])
vetor1 = vetor1.todense()
nomes1 = tf_idf.get_feature_names()
nomes1
df1 = pd.DataFrame(vetor1, columns = nomes1)
df1
texto2 = 'A matemática é incrível, quanto mais estudo matemática mais eu consigo aprender matemática'
tf_idf = TfidfVectorizer()
vetor2 = tf_idf.fit_transform([texto2])
vetor2 = vetor2.todense()
nomes2 = tf_idf.get_feature_names()
nomes2
df2 = pd.DataFrame(vetor2, columns = nomes2)
df2