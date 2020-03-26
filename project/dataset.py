import pandas as pd
import numpy as np

class Dataset:

    def __init__(self, name, max_len = 80):
        self.name = name
        self.df = None
        self.df_train = None
        self.df_test = None
        self.get_data()
        self.group_train = None
        self.group_test = None
        self.sentences_train = None
        self.sentences_test = None
        self.prepare_sentences()
        self.word_to_index = None
        self.index_to_word = None
        self.tag_to_index = None
        self.index_to_tag = None
        self.prepare_dictionaries()


    def get_data(self):
        # train = 'ner/%strain.bio' % self.name
        # test = 'ner/%stest.bio' % self.name
        train = '../ner/%strain.bio' % self.name
        test = '../ner/%stest.bio' % self.name
        self.df_train = self.read_data(train)
        self.df_test = self.read_data(test)
        self.df = pd.concat([self.df_train, self.df_test])

    def read_data(self, filename):
        data_dict = {'sentence_id': [], 'word': [], 'tag': []}
        with open(filename, 'r') as f:
            sentence_id = 0
            for line in f.readlines():
                if line == '\n':
                    sentence_id += 1
                    continue
                x = []
                line = line.strip().split()
                data_dict['sentence_id'].append(sentence_id)
                data_dict['word'].append(line[1])
                data_dict['tag'].append(line[0])
        return pd.DataFrame.from_dict(data_dict)


    def prepare_sentences(self):
        self.group_train = self.df_train.groupby('sentence_id', sort=False).apply(lambda s: [(w, t) for w, t in zip(s['word'].values.tolist(),s['tag'].values.tolist())])
        self.group_test = self.df_test.groupby('sentence_id', sort=False).apply(lambda s: [(w, t) for w, t in zip(s['word'].values.tolist(),s['tag'].values.tolist())])
        self.sentences_train = [s for s in self.group_train]
        self.sentences_test = [s for s in self.group_test]


    def prepare_dictionaries(self):
        df = self.df
        words = df['word'].unique()
        tags = df['tag'].unique()
        self.word_to_index = {'PADword': 0, 'UNKNOWNword': 1}
        self.index_to_word = {0: 'PADword', 1: 'UNKNOWNword'}
        for i, word in enumerate(words):
            self.word_to_index[word] = i+2
            self.index_to_word[i+2] = word
        self.tag_to_index = {'PADword': 0}
        self.index_to_tag = {0: 'PADword'}
        for i, tag in enumerate(tags):
            self.tag_to_index[tag] = i+1
            self.index_to_tag[i+1] = tag

    def get_sentences(self):
        return self.sentences_train, self.sentences_test

    def get_sentences_labels(self):
        ''' get sentences and labels separately'''

        # self.prepare_sentences()
        # self.df['label'] = self.df_train['tag'].map(lambda x: self.tag_to_index[x])
        # data = self.df.groupby('sentence_id', sort=False).agg({'word':' '.join, 'label': list})
        self.prepare_sentences()
        self.df_train['label'] = self.df_train['tag'].map(lambda x: self.tag_to_index[x])
        self.df_test['label'] = self.df_test['tag'].map(lambda x: self.tag_to_index[x])
        train = self.df_train.groupby('sentence_id', sort=False).agg({'word':' '.join, 'label': list})
        test = self.df_test.groupby('sentence_id', sort=False).agg({'word':' '.join, 'label': list})
        data = pd.concat([train, test])
        return data['word'], data['label']

    def split_train_test(self, data):
        train_len = self.df_train.sentence_id.max() + 1
        return data[: train_len], data[train_len: ]

    def get_sentences_labels_splits(self):
        ''' get sentences and labels separately for train and test'''

        self.prepare_sentences()
        self.df_train['label'] = self.df_train['tag'].map(lambda x: self.tag_to_index[x])
        self.df_test['label'] = self.df_test['tag'].map(lambda x: self.tag_to_index[x])
        train = self.df_train.groupby('sentence_id', sort=False).agg({'word':' '.join, 'label': list})
        test = self.df_test.groupby('sentence_id', sort=False).agg({'word':' '.join, 'label': list})

        return train['word'], train['label'], test['word'], test['label']

    def get_dictionaries(self):
        return self.word_to_index, self.index_to_word, self.tag_to_index, self.index_to_tag

    def save(self):
        '''save processed dataframe'''
        raise NotImplementedError
