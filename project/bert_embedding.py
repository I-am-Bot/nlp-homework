import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers
import pandas as pd
import numpy as np
import argparse
from dataset import Dataset
import numpy as np

import warnings
warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='restaurant', choices=['restaurant', 'eng'], help='dataset')

args = parser.parse_args()
data = Dataset(name=args.dataset)

# # get sentences samples, for LSTM, WORD2VEC, BERT...
X, y = data.get_sentences_labels()


# For DistilBERT:
model_class = transformers.DistilBertModel
tokenizer_class = transformers.DistilBertTokenizer
pretrained_weights = 'distilbert-base-uncased'

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized = X.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# calculate max length
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = ("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

input_ids = torch.tensor(padded).to(device)
attention_mask = torch.tensor(attention_mask).to(device)

model = model.to(device)
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:,0,:].numpy()


# TODO add LSTM, add CONLL
import ipdb
ipdb.set_trace()

