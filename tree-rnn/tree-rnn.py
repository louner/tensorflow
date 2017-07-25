import tensorflow as tf
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from pycorenlp import StanfordCoreNLP
from form_parse_tree import build_lexical_tree, embedding_size, WNJsonDecoder
from preprocess import load_data
import logging
import tensorflow_fold as td
import traceback
from read_data import make_batch
import json
from time import time

logging.basicConfig(filename='log/lab.log', level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger('lab')

weights = {}
similarity_w = tf.Variable(tf.random_normal([2, embedding_size*2]), name='simi_w')
similarity_b = tf.Variable(tf.random_normal([2, 1]), name='simi_b')

answers = [tf.constant(np.asarray([[1], [0]]), dtype=tf.float32), tf.constant(np.asarray([[0], [1]]), dtype=tf.float32)]
learning_rate = 0.0001
batch_size = 5000
epsilon = tf.constant(value=1e-5)
number_classes = 2

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
    loss = tf.reduce_sum(tf.multiply(label, tf.log(predict+1e-10), name='loss'))
    return loss

def optimie(loss):
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer.minimize(loss, var_list=[w, b])

def buid_sentence_expression():
    sentence_tree = td.InputTransform(lambda sentence_json: WNJsonDecoder(sentence_json))

    tree_rnn = td.ForwardDeclaration(td.PyObjectType())
    leaf_case = td.GetItem('word_vec', name='leaf_in') >> td.Vector(embedding_size)
    index_case = td.Record({'children': td.Map(tree_rnn()) >> td.Mean(), 'word_vec': td.Vector(embedding_size)}, name='index_in') >> td.Concat(name='concat_root_child') >> td.FC(embedding_size, name='FC_root_child')
    expr_sentence = td.OneOf(td.GetItem('leaf'), {True: leaf_case, False: index_case}, name='recur_in')
    tree_rnn.resolve_to(expr_sentence)

    return sentence_tree >> expr_sentence

def create_compiler(file_queue):
    expr_left_sentence, expr_right_sentence = buid_sentence_expression(), buid_sentence_expression()

    expr_label = td.InputTransform(lambda label: int(label)) >> td.OneHot(2, dtype=tf.float32)
    one_record = td.InputTransform(lambda record: json.loads(record.decode('utf-8'))) >> \
                 td.Record((expr_left_sentence, expr_right_sentence, expr_label), name='instance')
                 #td.AllOf(td.slice(start=0, stop=2) >> td.Concat() >> td.FC(2), td.Slice(start=2, stop=3))

    batch = make_batch(file_queue)
    compiler = td.Compiler().create(one_record, input_tensor=batch)
    return compiler

def build_graph(filequeue):
    compiler = create_compiler(file_queue)
    sentence1, sentence2, label_vector = compiler.output_tensors

    w = tf.Variable(tf.random_normal([embedding_size, number_classes]), name='similarity_w')
    b = tf.Variable(tf.random_normal([1, number_classes]), name='similarity_b')
    distance = tf.multiply(sentence1, sentence2)
    logist = tf.sigmoid(tf.matmul(distance, w) + b)

    loss = -1 * tf.reduce_mean(tf.log(logist + epsilon) * label_vector)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return logist, loss, train_step, w
'''
compiler = td.Compiler().create(one_record)
'''
if __name__ == '__main__':
    file_queue = tf.train.string_input_producer(['data/tree.%d'%(i) for i in range(10)])
    #file_queue = tf.train.string_input_producer(['data/tree.test'])

    logist, loss, train_step, w = build_graph(file_queue)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        writer = tf.summary.FileWriter('log', sess.graph)

        _, _, test_X, test_Y = load_data()

        init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()
        sess.run([init, local_init])

        st = time()
        i = 0
        while True:
            i += 1
            l, t, _ = sess.run([logist, loss, train_step])
            print(time()-st, list(l))
            if i % 40 == 0:
                saver.save(sess, 'models/model_%s' % (i))
                if i == 400:
                    break
        '''
        while st_index < len(train_X):
            train = [(inst[0], inst[1], label) for inst, label in zip(train_X[st_index:st_index+batch_size], train_Y[st_index:st_index+batch_size])]
            batch = compiler.build_feed_dict(train)
            try:
                sess.run([train_step], feed_dict=batch)
            except:
                logger.error(traceback.format_exc())
                continue

            writer = tf.summary.FileWriter('log', sess.graph)

            p = []
            for inst, label in zip(test_X, test_Y):
                predict = tf.argmax(predict_semantics_equality(inst[0], inst[1]), 0, name='predict')
                print('test', predict)
                p.append(predict)

            predicts = tf.cast(tf.stack(p), dtype=tf.int8)
            print(predicts, labels)

            accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(predicts, labels), dtype=tf.float32))

            sess.run(minimizers)
            print('accuracy %f'%(sess.run(accuracy)))

        '''
