from multiprocessing import Process
from form_parse_tree import build_lexical_tree
import pandas

shared = 'shared !'
df = pandas.read_csv('data/train.csv')

def build_trees(id, total):
    for i in range(len(df)):
        if i%total == id:
            instance = df.loc[i]
            sentence1, sentence2, label = instance['question1'], instance['question2'], instance['is_duplicate']
            r1, r2 = build_lexical_tree(sentence1), build_lexical_tree(sentence2)
            print(id, sentence1, sentence2)

numProcess = 400
processes = [Process(target=build_trees, args=(i, numProcess)) for i in range(numProcess)]
for process in processes:
    process.start()

for process in processes:
    process.join()