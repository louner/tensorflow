import tensorflow as tf
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from pycorenlp import StanfordCoreNLP
from form_parse_tree import build_lexical_tree, embedding_size
from preprocess import load_data
import logging
logger = logging.getLogger()

import gc
gc.enable()

weights = {}
similarity_w = tf.Variable(tf.random_normal([2, embedding_size*2]), name='simi_w')
similarity_b = tf.Variable(tf.random_normal([2, 1]), name='simi_b')

answers = [np.asarray([1, 0]), np.asarray([0, 1])]
learning_rate = 0.0001

def add_weight(relation):
    w = tf.Variable(tf.random_normal([embedding_size, embedding_size*2]), name='%s_w'%(relation))
    b = tf.Variable(tf.random_normal([embedding_size, 1]), name='%s_b'%(relation))
    weights[relation] = [w, b]

def build_tree_rnn_graph(root):
    if root.is_leaf():
        return

    root.visited = True
    child_layers = []
    for relation, children in root.children.items():
        relation = relation.replace(':', '-')
        if not relation in weights:
            print(relation)
            add_weight(relation)

        W, B = weights[relation]
        for child in children:
            # if visited(cycle occur), stop trace in
            if not child.visited:
                build_tree_rnn_graph(child)

            concat_words = tf.concat([root.word_vec, child.hidden], axis=0)
            concat_words = tf.reshape(concat_words, [embedding_size*2, 1])
            child_layer = tf.matmul(W, concat_words) + B
            child_layers.append(child_layer)

    # is there a better way to merge tensors from children ?
    root.hidden = tf.reduce_mean(child_layers, axis=0)
    return

def sentence_graph(sentence):
    # tree whose node has word, lexical label
    lex_tree_root = build_lexical_tree(sentence)

    # build tree-rnn graph from lexical tree

    build_tree_rnn_graph(lex_tree_root)
    return lex_tree_root.hidden

def predict_semantics_equality(sentence1, sentence2):
    semantics1 = sentence_graph(sentence1)
    semantics2 = sentence_graph(sentence2)
    predict = tf.nn.softmax(tf.matmul(similarity_w, tf.concat([semantics1, semantics2], axis=0)) + similarity_b)

    return predict

def count_loss(sentence1, sentence2, label):
    predict = predict_semantics_equality(sentence1, sentence2)
    label = answers[label]
    loss = tf.reduce_sum(label * tf.log(predict+1e-10))
    return loss

def optimie(loss):
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    return optimizer.minimize(loss)

'''
s1 = 'Why are so many Quora users posting questions that are readily answered on Google?'
s2 = 'Why do people ask Quora questions which can be answered easily by Google?'
label = 1

loss = count_loss(s1, s2, label)
minimizer = optimie(loss)
'''

saver = tf.train.Saver()

with tf.Session() as sess:
    train_X, train_Y, test_X, test_Y = load_data()
    batch_size = 100
    st_index = 0

    init = tf.global_variables_initializer()
    sess.run(init)

    while st_index < len(train_X):
        minimizers, losses, initialized_vars = [], [], tf.global_variables()
        for inst, label in zip(train_X[st_index:st_index+batch_size], train_Y[st_index:st_index+batch_size]):
            loss = count_loss(inst[0], inst[1], label)
            minimizer = optimie(loss)
            minimizers.append(minimizer)
            #losses.append(loss)

        uninitialized_vars = list(set(tf.global_variables()) - set(initialized_vars))
        sess.run(tf.variables_initializer(uninitialized_vars))

        sess.run(minimizers)

        print(st_index, 'done')
        saver.save(sess, 'models/model_%s' % (st_index))
        st_index += batch_size
