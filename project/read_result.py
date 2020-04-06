

tag_to_index = {}

def read(filename):
    # (word, tag, prediction)
    res = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if len(line.split()) < 2:
                continue
            line = line.strip()
            res.append(line.split())
    return res

read('checkpoints/feature_bert/25.P0.94_R0.92_F0.93')
read('checkpoints/feature_bert/17.P0.94_R0.91_F0.92')
read('checkpoints/feature_bert/17.P0.94_R0.91_F0.92')

