from pycorenlp import StanfordCoreNLP
import json
import numpy as np
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
word2vec_model_filepath = '../word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin'
w2v = KeyedVectors.load_word2vec_format(word2vec_model_filepath, binary=True)
embedding_size = 300
unk = np.zeros([embedding_size, 1]).astype(np.float32)

def get_word_vector(word):
    try:
        return np.reshape(w2v.word_vec(word).astype(np.float32), [embedding_size, 1])
    except:
        return unk

nlp = StanfordCoreNLP('http://localhost:9000')

class WordNode:
    def __init__(self, word):
        self.word = word
        self.word_vec = get_word_vector(word)
        self.hidden = self.word_vec
        self.children = defaultdict(lambda: [], {})

        # dep parsing may have cycle, it's not a tree
        self.visited = False

    def add_child(self, node, relation):
        self.children[relation].append(node)

    def __repr__(self):
        return self.word

    def is_leaf(self):
        return not self.children

def trace(root):
    print(root.word)
    for rel in root.children:
        print(rel, root.children[rel])

    for rel in root.children:
        for node in root.children[rel]:
            trace(node)

def build_lexical_tree(sentence):
    output = nlp.annotate(sentence, properties={'annotators': 'parse', 'outputFormat': 'json'})['sentences'][0]
    #output = json.loads(json.dumps(output).lower())

    words = dict([(tok['word'], WordNode(tok['word'])) for tok in output['tokens']])
    words['ROOT'] = WordNode('ROOT')
    deps = output['basicDependencies']
    #deps = output['enhancedDependencies']

    for dep in deps:
        label, parent, child = dep['dep'].lower(), dep['governorGloss'], dep['dependentGloss']
        words[parent].add_child(words[child], label)

    return words['ROOT']

if __name__ == '__main__':
    sentence = 'Why are so many questions posted to Quora that are so easily answered by using Google?'
    print(trace(build_lexical_tree(sentence)))
    '''
    sentence = 'How can I win her back ?'
    sentence = 'Peter and I like apples .'
    print(trace(build_lexical_tree(sentence)))
    '''