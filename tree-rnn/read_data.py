import tensorflow as tf
import json

file_queue = tf.train.string_input_producer(['data.json'])

def read(file_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)
    return [value]

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    value = read(file_queue)

    print(sess.run(value))

    readers = [read(file_queue) for _ in range(5)]
    batch_size = 2
    capacity = 5
    min_after_dequeue = 2

    value_batch = tf.train.shuffle_batch_join(readers,
                                batch_size=batch_size,
                                capacity=capacity,
                                min_after_dequeue=min_after_dequeue)

    print(value_batch)
    print(sess.run(value_batch))

