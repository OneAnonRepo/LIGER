import numpy as np
import tensorflow as tf
import Queue
import threading

SMALL_NUMBER = 1e-7


def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)


class ThreadedIterator:
    """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator, max_queue_size):
        self.__queue = Queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()






class MLP(object):
    def __init__(self, inputs, hid_sizes, out_size, dropout_keep_prob):
        self.inputs = inputs
        self.hid_sizes = hid_sizes
        self.out_size = out_size
        self.dropout_keep_prob = dropout_keep_prob

    def __call__(self):
        if(self.hid_sizes == 0):
            output_layer = tf.layers.dense(self.inputs, self.out_size, activation=tf.nn.relu, trainable=True, name="output_layer")
            outputs = tf.layers.dropout(output_layer, self.dropout_keep_prob, name="outputs")            
            return outputs
        else:
            hidden_layer = tf.layers.dense(self.inputs, self.hid_sizes, activation=tf.nn.relu, trainable=True, name="hidden_layer")
            hidden_layer_dropout = tf.layers.dropout(hidden_layer, self.dropout_keep_prob, name="hidden_layer_dropout")
            output_layer = tf.layers.dense(hidden_layer_dropout, self.out_size, activation=tf.nn.relu, trainable=True, name="output_layer")
            outputs = tf.layers.dropout(output_layer, self.dropout_keep_prob, name="outputs")            
            return outputs
            
            
            
            
            
    