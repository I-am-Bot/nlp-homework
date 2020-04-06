import tensorflow_hub as hub
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from keras_contrib.layers import CRF as CRF_2nd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from sklearn_crfsuite import metrics
import torch
import transformers
import pandas as pd
import numpy as np
import argparse
from dataset import Dataset
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='restaurant', choices=['restaurant', 'eng'], help='dataset')

args = parser.parse_args()
data = Dataset(name=args.dataset)

# # get sentences samples, for LSTM, WORD2VEC, BERT...
X, y = data.get_sentences_labels()
_, y_test = data.split_train_test(y)

# # calculate max length
max_len = 1
# for i in X.values:
#     if len(i) > max_len:
#         max_len = len(i)

# padded = np.array([i + [0]*(max_len-len(i)) for i in X.values])

elmo = hub.Module("pretrained", trainable=False)
embeddings = elmo(
        X.values,
        signature="default",
        as_dict=True)["elmo"]

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    features = session.run(embeddings)

tag_to_index = data.tag_to_index
index_to_tag = data.index_to_tag
y = pad_sequences(maxlen=max_len, sequences=y.values, padding="post", value=tag_to_index["PADword"])
y = [to_categorical(i, num_classes=len(tag_to_index)) for i in y]

X_train, X_test = data.split_train_test(features)
y_train, _ = data.split_train_test(y)

input = Input(shape = (features.shape[1], features.shape[2],))
model = Bidirectional(LSTM(units = 50, return_sequences=True, recurrent_dropout=0.1))(input)
model = TimeDistributed(Dense(50, activation="relu"))(model)
crf = CRF_2nd(len(data.tag_to_index))
out_layer = crf(model)

model = Model(input, out_layer)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()
BATCH_SIZE = 64
EPOCHS = 10
history = model.fit(X_train, np.array(y_train), batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=0.1, verbose=2)

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PADword", "O"))
        out.append(out_i)
    return out

pred_y = model.predict(X_test)
pred_y = np.argmax(pred_y, axis=-1)
y_test_true = np.argmax(y_test, -1)

pred_y = [[index_to_tag[i].replace("PADword", "O") for i in pred_y[index]][0: len(y_test[index])] for index in range(len(pred_y))]
y_test_true = [[index_to_tag[i] for i in row] for row in y_test]

print('LSTM Classification Report\n', metrics.flat_classification_report(pred_y, y_test_true))

