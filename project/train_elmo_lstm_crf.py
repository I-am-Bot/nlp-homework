import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from models.elmo_lstm_crf import Net
from dataset import ELMoDataset, pad_elmo, VOCAB, tokenizer, tag2idx, idx2tag
import os
import numpy as np
import argparse
import tensorflow_hub as hub
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--finetuning", dest="finetuning", action="store_true")
parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
parser.add_argument("--logdir", type=str, default="checkpoints/feature_elmo_crf")
parser.add_argument("--trainset", type=str, default="Data/Conll2003_NER/train.txt")
parser.add_argument("--validset", type=str, default="Data/Conll2003_NER/valid.txt")
hp = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


elmo = hub.Module("pretrained", trainable=False)
def train(model, iterator, optimizer, criterion, device):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch

        x = elmo(
                inputs={'tokens':x, 'sequence_len': [len(x[0])]*len(x)},
                signature="tokens",
                as_dict=True)["elmo"]

        config = tf.ConfigProto(device_count = {'GPU': 0})
        with tf.Session(config=config) as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            x = session.run(x)
        x = torch.FloatTensor(x)
        x = x.to(device)
        y = y.to(device)
        _y = y # for monitoring
        optimizer.zero_grad()

        loss = model.neg_log_likelihood(x, y) # logits: (N, T, VOCAB), y: (N, T)
        loss.backward()
        optimizer.step()

        if i%10==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")

def eval(model, iterator, f, device):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            x = elmo(
                    inputs={'tokens':x, 'sequence_len': [len(x[0])]*len(x)},
                    signature="tokens",
                    as_dict=True)["elmo"]

            config = tf.ConfigProto(device_count = {'GPU': 0})
            with tf.Session(config=config) as session:
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                x = session.run(x)
            x = torch.FloatTensor(x)

            x = x.to(device)
            # y = y.to(device)

            _, y_hat = model(x)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open("temp_elmo_crf", 'w', encoding='utf-8') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]

            assert len(preds)==len(words.split())==len(tags.split())
            # assert len(preds)==len(words)==len(tags.split())
            for w, t, p in zip(words.split(), tags.split(), preds):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true =  np.array([tag2idx[line.split()[1]] for line in open("temp_elmo_crf", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open("temp_elmo_crf", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    final = f + ".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
    with open(final, 'w', encoding='utf-8') as fout:
        result = open("temp_elmo_crf", "r", encoding='utf-8').read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")
    os.remove("temp_elmo_crf")
    print("precision=%.4f"%precision)
    print("recall=%.4f"%recall)
    print("f1=%.4f"%f1)
    return precision, recall, f1

tag2idx['[CLS]'] = len(tag2idx)
tag2idx['[SEP]'] = len(tag2idx)
idx2tag[tag2idx['[CLS]']] = '[CLS]'
idx2tag[tag2idx['[SEP]']] = '[SEP]'
model = Net(tag2idx, device=device, finetuning=hp.finetuning).cuda()

train_dataset = ELMoDataset(hp.trainset)
eval_dataset = ELMoDataset(hp.validset)

train_iter = data.DataLoader(dataset=train_dataset,
                             batch_size=hp.batch_size,
                             shuffle=True,
                             num_workers=4,
                             collate_fn=pad_elmo)
eval_iter = data.DataLoader(dataset=eval_dataset,
                             batch_size=hp.batch_size,
                             shuffle=False,
                             num_workers=4,
                             collate_fn=pad_elmo)

optimizer = optim.Adam(model.parameters(), lr = hp.lr)
criterion = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(1, hp.n_epochs+1):
    train(model, train_iter, optimizer, criterion, device)
    print(f"=========eval at epoch={epoch}=========")
    if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
    fname = os.path.join(hp.logdir, str(epoch))
    precision, recall, f1 = eval(model, eval_iter, fname, device)

    torch.save(model.state_dict(), f"{fname}.pt")
    print(f"weights were saved to {fname}.pt")

