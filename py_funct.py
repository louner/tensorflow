import numpy as np
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors
w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
MAX_SENTENCE_LENGTH = 20
BATCH_SIZE = 2

filename_queue = tf.train.string_input_producer(['input.csv'])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

label, sentence = tf.decode_csv(value, record_defaults=[[1], ['string']], field_delim=',')

def split_add_prefix(ary):
    sentence = '-'. join(ary.decode('utf-8').split(' '))
    return str.encode(sentence)

def transform_to_w2v_matrix(sentence):
    toks = sentence.decode('utf-8').lower().split(' ')
    matrix = []
    for tok in toks:
        try:
            vector = w2v.word_vec(tok)
            matrix.append(vector)
        except:
            pass
    if len(matrix) < MAX_SENTENCE_LENGTH:
        padding = [np.zeros([300]) for _ in range(MAX_SENTENCE_LENGTH-len(matrix))]
        matrix += padding

    elif len(matrix) > MAX_SENTENCE_LENGTH:
        matrix = matrix[:MAX_SENTENCE_LENGTH]

    return np.asarray(matrix)

splitt = tf.py_func(split_add_prefix, [sentence], tf.string)
w2v_matrix = tf.py_func(transform_to_w2v_matrix, [sentence], tf.double)
batch_x = tf.train.shuffle_batch([splitt], batch_size=BATCH_SIZE, capacity=BATCH_SIZE*3, min_after_dequeue=BATCH_SIZE*3)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    ret = sess.run(batch_x)