from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn import decomposition
import numpy as np

class perceptron():
    def __init__(self, dim):
        self.learning_rate = 10
        self.max_iteration = 200
        self.dim = dim
        self.w = np.random.random([20, dim + 1])

    def train(self, feature, labels):
        for k in range(self.max_iteration):
            if (k == 50):
                self.learning_rate = 1
            if (k == 100):
                self.learning_rate = 0.1

            for i in range(np.size(feature, 0)):
                vector = feature[i]
                vector = np.insert(vector, self.dim, 1)
                label = labels[i]
                # print(label)
            
                # print(vector.shape)
                # print(self.w.shape)

                pred = self.w @ vector
                # print(pred)
                pred = self.sigmoid(pred)
                # print(pred)
                pred = pred.reshape(20,1)
                # print(pred)
                label_onehot = self.onehot(label)
                l = self.loss(label_onehot, pred)
                # print(l)
                # print(self.w)
                self.w = self.w + self.learning_rate * np.multiply( (label_onehot - pred), np.multiply(pred, (1 - pred)) ) @  vector.reshape(1, self.dim + 1)
                # a = input() 
            print(l)
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
            # print(pred) 
            
            pred = self.sigmoid(pred)
            labels.append(np.argmax(pred))
        return labels

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def loss(self, label, pred):
        return np.mean(np.square(label - pred))



newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42,remove = ('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42,remove = ('headers', 'footers', 'quotes'))

vectorizer = TfidfVectorizer(stop_words='english',lowercase=True)
vectorizer.fit(newsgroups_train.data)
train_vector = vectorizer.transform(newsgroups_train.data)
test_vector = vectorizer.transform(newsgroups_test.data)

decomp =  decomposition.TruncatedSVD(n_components = 300)
train_pca = decomp.fit_transform(train_vector)
test_pca = decomp.transform(test_vector)




from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

newsgroups_test = fetch_20newsgroups(subset='test')

per = perceptron(dim = 300)
per.train(train_pca, newsgroups_train.target)
pred = per.predict(test_pca)

# print(pred)
print(metrics.accuracy_score(newsgroups_test.target,pred))