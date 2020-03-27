import pandas as pd
import numpy as np
import argparse
from dataset import Dataset
import numpy as np
from featureextractor import sent2features, sent2labels
from keras_contrib.layers import CRF as CRF_2nd
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.models import Model, Input
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import preprocessing
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter

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
print('Perceptron Classification Report\n', classification_report(y_pred=per.predict(X_test), y_true=y_test))
# per.partial_fit(X_train, y_train, classes)
# print(classification_report(y_pred=per.predict(X_test), y_true=y_test, labels=new_classes))

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
print('SGDClassifier Classification Report\n', classification_report(y_pred=sgd.predict(X_test), y_true=y_test))
# sgd.partial_fit(X_train, y_train, classes)
# print(classification_report(y_pred=per.predict(X_test), y_true=y_test, labels=new_classes))

nb = MultinomialNB(alpha=0.01)
nb.fit(X_train, y_train)
print('MultinomialNB Classification Report\n', classification_report(y_pred=nb.predict(X_test), y_true=y_test))
# nb.partial_fit(X_train, y_train, classes)
# print(classification_report(y_pred=per.predict(X_test), y_true=y_test, labels=new_classes))

pa = PassiveAggressiveClassifier()
pa.fit(X_train, y_train)
print('PassiveAggressiveClassifier Classification Report\n', classification_report(y_pred=pa.predict(X_test), y_true=y_test))
# pa.partial_fit(X_train, y_train, classes)
# print(classification_report(y_pred=per.predict(X_test), y_true=y_test, labels=new_classes))


# get sentences samples, for CRF, LSTM, WORD2VEC, BERT...
train, test = data.get_sentences()
train_X = [sent2features(s) for s in train]
train_Y = [sent2labels(s) for s in train]
test_X = [sent2features(s) for s in test]
test_Y = [sent2labels(s) for s in test]


crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True

)
crf.fit(train_X, train_Y)

pred_Y = crf.predict(test_X)
print('CRF Classification Report\n', metrics.flat_classification_report(pred_Y, test_Y))

# Bidirectional LSTM with CRF
# hyper parameters
BATCH_SIZE = 200
EPOCHS = 100
MAX_LEN = 25
EMBEDDING = 40

# checkpoints = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose = 0, mode = 'auto', save_best_only = True, monitor='val_loss')

word_to_index, index_to_word, tag_to_index, index_to_tag = data.get_dictionaries()

#total_height = tf.cast(tf.convert_to_tensor(ymax_int - ymin_int, dtype = tf.float32), dtype = tf.int32)
no_tags = len(tag_to_index)
no_words = len(word_to_index)


train_X = [[word_to_index[w[0]] for w in s] for s in train]
train_X = pad_sequences(maxlen = MAX_LEN, sequences = train_X, padding = "post", value = word_to_index["PADword"])
train_y = [[tag_to_index[w[1]] for w in s] for s in train]
train_y = pad_sequences(maxlen = MAX_LEN, sequences = train_y, padding = "post", value = tag_to_index["PADword"])
train_y = [to_categorical(i, num_classes=no_tags) for i in train_y]

test_X = [[word_to_index[w[0]] for w in s] for s in test]
test_X = pad_sequences(maxlen = MAX_LEN, sequences = test_X, padding = "post", value = word_to_index["PADword"])
test_y = [[tag_to_index[w[1]] for w in s] for s in test]
test_y = pad_sequences(maxlen = MAX_LEN, sequences = test_y, padding = "post", value = tag_to_index["PADword"])
test_y = [to_categorical(i, num_classes=no_tags) for i in test_y]

input = Input(shape = (MAX_LEN,))

model = Embedding(input_dim = no_words, output_dim = EMBEDDING, input_length = MAX_LEN, mask_zero=True)(input)
model = Bidirectional(LSTM(units = 50, return_sequences=True, recurrent_dropout=0.1))(model)
model = TimeDistributed(Dense(50, activation="relu"))(model)
crf = CRF_2nd(no_tags)
out_layer = crf(model)

model = Model(input, out_layer)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()

history = model.fit(train_X, np.array(train_y), batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=0.1, verbose=2)

pred_y = model.predict(test_X)
pred_y = np.argmax(pred_y, axis=-1)
y_test_true = np.argmax(test_y, -1)
y_test_true = [[index_to_tag[i] for i in row] for row in y_test_true]
y_test_true = [[x for x in row if x!='PADword'] for row in y_test_true]

pred_y = [[index_to_tag[i] for i in row] for row in pred_y]
pred_y = [[x.replace("PADword", "O") for x in pred_y[index]][: len(y_test_true[index])] for index in range(len(y_test_true))]

print('LSTM Classification Report\n', metrics.flat_classification_report(pred_y, y_test_true))

