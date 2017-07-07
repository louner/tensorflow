from pycorenlp import StanfordCoreNLP
import json
from json import JSONEncoder, JSONDecoder
import numpy as np
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
import re
import logging
from preprocess import load_data
word2vec_model_filepath = '../word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin'
w2v = KeyedVectors.load_word2vec_format(word2vec_model_filepath, binary=True)
embedding_size = 300
unk = np.zeros([embedding_size]).astype(np.float32).tolist()
wordPat = re.compile(r'[^0-9a-zA-Z]')

logging.basicConfig(filename='log/form_parse_tree.log', level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger('form_parse_tree')

def get_word_vector(word):
    try:
        #return np.reshape(w2v.word_vec(word).astype(np.float32), [embedding_size, 1])
        return w2v.word_vec(word).astype(np.float32).tolist()
        #return np.ones([3]).tolist()
    except:
        return unk

nlp = StanfordCoreNLP('http://localhost:9000')

class WordNode:
    def __init__(self, word=None):
        if word:
            self.word = wordPat.sub('X', word)
            self.word_vec = get_word_vector(word)
            self.hidden = self.word_vec
            #self.children = defaultdict(lambda: [], {})
            self.children = []

            # dep parsing may have cycle, it's not a tree
            self.visited = False

    def add_child(self, node, relation):
        #self.children[relation].append(node)
        self.children.append(node)

    def __repr__(self):
        return self.word

    def is_leaf(self):
        return not self.children


def WNJsonEncoder(node):
    #print(node)
    children = [WNJsonEncoder(child) for child in node.children]
    return {'word': node.word, 'word_vec': node.word_vec, 'children':children}

def WNJsonDecoder(nodeJson):
    root = WordNode()
    root.word = nodeJson['word']
    root.word_vec = nodeJson['word_vec']
    root.hidden = root.word_vec
    children = [WNJsonDecoder(child) for child in nodeJson['children']]
    root.children = children
    return root

error_tree = WordNode('error')

def trace(root):
    print(root.word, root.leaf)
    for child in root.children:
        print(root.word, '>', child.word)

    for child in root.children:
        trace(child)

def build_lexical_tree(sentence):
    try:
        output = nlp.annotate(sentence, properties={'annotators': 'parse', 'outputFormat': 'json'})['sentences'][0]
    except:
        logger.error(sentence)
        return error_tree
    #output = json.loads(json.dumps(output).lower())

    words = dict([(tok['word'], WordNode(tok['word'])) for tok in output['tokens']])
    words['ROOT'] = WordNode('ROOT')
    deps = output['basicDependencies']
    #deps = output['enhancedDependencies']

    for dep in deps:
        label, parent, child = dep['dep'].lower(), dep['governorGloss'], dep['dependentGloss']
        words[parent].add_child(words[child], label)

    remove_cycle(words['ROOT'])

    return words['ROOT']

def remove_cycle(root):
    toExpand = [root]
    while toExpand:
        nextToExpand = []
        for node in toExpand:
            node.visited = True
            children = [child for child in node.children if not child.visited]
            node.children = children
            node.leaf = not node.children
            nextToExpand += children
        toExpand = list(set(nextToExpand))

if __name__ == '__main__':
    import traceback
    train_X, train_Y, test_X, test_Y = load_data()
    f = open('data.json', 'w')
    batch, batch_size = [], 500
    for inst, label in zip(train_X, train_Y):
        try:
            batch.append([WNJsonEncoder(build_lexical_tree(inst[0])), WNJsonEncoder(build_lexical_tree(inst[1])), str(label)])
        except:
            logger.info(traceback.format_exc())
            break
    print(batch[0])
    json.dump(batch, f)
    f.close()
    '''
    sentence = 'How can I win her back ?'
    sentence = 'Peter and I like apples .'
    print(trace(build_lexical_tree(sentence)))
    '''