from multiprocessing import Process
from form_parse_tree import build_lexical_tree, WNJsonDecoder, WNJsonEncoder
import pandas
import json

shared = 'shared !'
df = pandas.read_csv('data/train.csv')

def build_trees(id, total):
    f = open('data/tree.%d'%(id), 'w')
    for i in range(len(df)):
        if i%total == id:
            instance = df.loc[i]
            sentence1, sentence2, label = instance['question1'], instance['question2'], instance['is_duplicate']
            r1, r2 = build_lexical_tree(sentence1), build_lexical_tree(sentence2)
            r1, r2 = WNJsonEncoder(r1), WNJsonEncoder(r2)
            f.write('%s\n'%(json.dumps([r1, r2, str(label)])))
    f.close()

def build_tree_in_parallel(numProcess=10):
    processes = [Process(target=build_trees, args=(i, numProcess)) for i in range(numProcess)]
    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    build_tree_in_parallel()