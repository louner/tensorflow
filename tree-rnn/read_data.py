import tensorflow as tf
import json

def read(file_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)
    return [value]

def make_batch(file_queue):
    readers = [read(file_queue) for _ in range(5)]
    batch_size = 10000
    min_after_dequeue = batch_size
    capacity = min_after_dequeue*3 + batch_size

    value_batch = tf.train.shuffle_batch_join(readers,
                                batch_size=batch_size,
                                capacity=capacity,
                                min_after_dequeue=min_after_dequeue)

    return value_batch

if __name__ == '__main__':
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        '''
        value = read(file_queue)
        print(sess.run(value))
        '''
        value = make_batch(file_queue)
        print(sess.run(value))
