from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn import decomposition
import numpy as np

class perceptron():
    def __init__(self, dim):
        self.learning_rate = 0.01
        self.max_iteration = 200
        self.dim = dim
        self.w = np.zeros([20, dim + 1])

    def train(self, feature, labels):

        for i in range(np.size(feature, 0)):
            vector = feature[i]
            vector = np.insert(vector, self.dim, 1)
            label = labels[i]
            # print(self.w.shape)

            pred = self.w @ vector
            pred = self.sigmoid(pred)
            pred = pred.reshape(20,1)
            print(pred)
            l = self.loss(self.onehot(label), pred)
            # print(l)
            # print(self.w)
            self.w = self.w + self.learning_rate * np.multiply( (label - pred), np.multiply(pred, (1 - pred)) ) * vector.reshape(1, self.dim + 1)
            a = input()

    def onehot(self, label):
        one_hot = np.zeros(20)
        one_hot[label] = 1
        return one_hot.reshape(20,1)

    def predict(self, features):
        labels = []
        for i in range(np.size(features, 0)):
            vector = features[i]
            vector = np.insert(vector, self.dim, 1)
            pred = self.w @ vector
            print(pred) 
            
            pred = self.sigmoid(pred)
            labels.append(np.argmax(pred))
        return labels

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def loss(self, label, pred):
        return np.mean(np.square(label - pred))

newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42,remove = ('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42,remove = ('headers', 'footers', 'quotes'))

vectorizer = TfidfVectorizer(stop_words='english',lowercase=True)
train_vector = vectorizer.fit_transform(newsgroups_train.data)
test_vector = vectorizer.transform(newsgroups_test.data)

train_pca = decomposition.TruncatedSVD(n_components = 300).fit_transform(train_vector)
test_pca = decomposition.TruncatedSVD(n_components = 300).fit_transform(test_vector)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

newsgroups_test = fetch_20newsgroups(subset='test')

per = perceptron(dim = 300)
per.train(train_pca, newsgroups_train.target)
pred = per.predict(test_pca)

print(pred)
print(metrics.f1_score(newsgroups_test.target, pred, average='macro'))

