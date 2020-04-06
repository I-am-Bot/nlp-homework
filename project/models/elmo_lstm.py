'''Modified based on https://github.com/Kyubyong/bert_ner/blob/master/model.py'''
import torch
import torch.nn as nn
import tensorflow_hub as hub
import tensorflow as tf
from pytorch_pretrained_bert import BertModel

class Net(nn.Module):
    def __init__(self, top_rnns=False, vocab_size=None, device='cpu', finetuning=False):
        super().__init__()

        emb_size = 1024
        self.top_rnns=top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=emb_size, hidden_size=emb_size//2, batch_first=True)

        self.fc = nn.Linear(emb_size, vocab_size)

        self.device = device
        self.finetuning = finetuning
        self.elmo = hub.Module("pretrained", trainable=False)

    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''

        # x = self.elmo(
        #         inputs={'tokens':x, 'sequence_len': [len(x[0])]*len(x)},
        #         signature="tokens",
        #         as_dict=True)["elmo"]

        # config = tf.ConfigProto(device_count = {'GPU': 0})
        # with tf.Session(config=config) as session:
        #     session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        #     enc = session.run(x)

        # enc = torch.FloatTensor(enc).to(self.device)

        enc = x.to(self.device)
        # enc = torch.FloatTensor(x).to(self.device)

        y = y.to(self.device)
        enc, _ = self.rnn(enc)
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

