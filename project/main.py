import pandas as pd
import numpy as np
import argparse
from dataset import Dataset
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='restaurant', choices=['restaurant', 'eng'], help='dataset')

args = parser.parse_args()
data = Dataset(name=args.dataset)
train, test = data.df_train, data.df_test

# # get sentences samples, for LSTM, WORD2VEC, BERT...
# train, test = data.get_sentences()

print(train.head())
print('All Tags:', train.tag.unique())

v = DictVectorizer(sparse=True)
y_train = train.tag.values
X_train = train.drop('tag', 1)
X_train = v.fit_transform(X_train.to_dict('records'))

y_test = test.tag.values
X_test = test.drop('tag', 1)
X_test = v.transform(X_test.to_dict('records'))

classes = np.unique(y_train).tolist()
new_classes = np.unique(y_train).tolist()

# label_encoder = preprocessing.LabelEncoder()
# y_train = label_encoder.fit_transform(y_train)
# y_train = label_encoder.fit(y_train)


per = Perceptron(n_jobs=-1, max_iter=100)
per.fit(X_train, y_train)
print(classification_report(y_pred=per.predict(X_test), y_true=y_test))
# per.partial_fit(X_train, y_train, classes)
# print(classification_report(y_pred=per.predict(X_test), y_true=y_test, labels=new_classes))

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
print(classification_report(y_pred=sgd.predict(X_test), y_true=y_test))
# sgd.partial_fit(X_train, y_train, classes)
# print(classification_report(y_pred=per.predict(X_test), y_true=y_test, labels=new_classes))

nb = MultinomialNB(alpha=0.01)
nb.fit(X_train, y_train)
print(classification_report(y_pred=nb.predict(X_test), y_true=y_test))
# nb.partial_fit(X_train, y_train, classes)
# print(classification_report(y_pred=per.predict(X_test), y_true=y_test, labels=new_classes))

pa = PassiveAggressiveClassifier()
pa.fit(X_train, y_train)
print(classification_report(y_pred=pa.predict(X_test), y_true=y_test))
# pa.partial_fit(X_train, y_train, classes)
# print(classification_report(y_pred=per.predict(X_test), y_true=y_test, labels=new_classes))


# TODO apply CRF
# get sentences samples, for CRF, LSTM, WORD2VEC, BERT...
train, test = data.get_sentences()



