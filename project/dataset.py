import pandas as pd

class Dataset:

    def __init__(self, name):
        self.name = name
        self.df_train = None
        self.df_test = None
        self.get_data()

    def get_data(self):
        train = '../ner/%strain.bio' % self.name
        test = '../ner/%stest.bio' % self.name
        self.df_train = self.read_data(train)
        self.df_test = self.read_data(test)

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

    def get_sentences(self):
        return self.df_train.groupby('sentence_id', sort=False).agg({'word':' '.join, 'tag': ' '.join}), self.df_test.groupby('sentence_id', sort=False).agg({'word':' '.join, 'tag': ' '.join})

    def save(self):
        '''save processed dataframe'''
        raise NotImplementedError
