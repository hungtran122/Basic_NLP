from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import platform
from nltk.corpus import stopwords
import time
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import nltk
nltk.download('stopwords')
nltk.download('punkt')
special_marks = [".",",",";","!","'",":"]
class config:
    def __init__(self):
        self.stopwords = stopwords.words("english")
        for mark in special_marks:
            self.stopwords.append(mark)
        self.vocabulary_size = 200
        if platform.system() == "Linux":
            self.filename = os.getcwd() + "/Data/LoR_new"
        else:
            self.filename = os.getcwd() + "\\data\\LoR_new"
        self.batch_size = 12
        self.embedding_size = 128  # Dimension of the embedding vector.
        self.skip_window = 1       # How many words to consider left and right.
        self.num_skips = 2        # How many times to reuse an input to generate a label.
        self.num_sampled = 64
class w2v_model():
    def __init__(self):
        self.config = config()
        self.add_placeholder()
        self.embedding = self.add_embedding()
        self.loss = self.add_loss_op()
        self.train = self.add_train_op()
        # Read the data into a list of strings.
    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(dtype=tf.int32,shape=[self.config.batch_size])
        self.label_placeholder = tf.placeholder(shape=[self.config.batch_size,None],dtype=tf.int32)
        print(tf.shape(self.input_placeholder))
    def add_embedding(self):
        embedding = tf.Variable(tf.random_uniform([self.config.batch_size,self.config.embedding_size],-1.0, 1.0))
        return tf.nn.embedding_lookup(embedding, self.input_placeholder)
    def add_loss_op(self):
        nce_weights = tf.Variable(tf.truncated_normal([self.config.batch_size, self.config.embedding_size],
                                                      stddev=1.0/math.sqrt(self.config.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.config.batch_size]))
        return tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                             biases = nce_biases,
                                             inputs = self.embedding,
                                             labels = self.label_placeholder,
                                             num_sampled = self.config.num_sampled,
                                             num_classes = self.config.batch_size))
    def add_train_op(self):
        return tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)
    def read_data(self):
        processed_word_list = []
        start = time.time()
        with open(self.config.filename, 'r', encoding="utf-8") as f:
            data = f.read()
            # for line in f:
            for word in nltk.word_tokenize(data):
                word = word.lower() # in case they arenet all lower cased
                if word not in self.config.stopwords:
                    processed_word_list.append(word)
        print("Read data took {:.2f} second".format(time.time()-start))
        print("Size of vocabulary", len(processed_word_list), "including recurrence")
        #print ("Stop words contains: ", self.config.stopwords)
        return processed_word_list
    def build_dataset(self, words): #words = processed word list
        """Process raw inputs into a dataset."""
        count = [['UNK', -1]] # count is a list, each element is a list
        # print("count = ",type(count),count[0][0])
        # collect most common word, after this count size becomes n_words (50000)
        count.extend(collections.Counter(words).most_common(self.config.vocabulary_size - 1))
        dictionary = dict()
        #each element in count has a word and occurences
        #store in dictionary with each word and its key
        #ex: UNK - 0, the - 1, of - 2, and - 3, one - 4, in - 5
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        i = 0
        #words is all word from training data with lenth 17005207
        #dictionary is a dict with length 50000
#        for word in words:
#            if word in dictionary:
#                index = dictionary[word]
#            else:
#                index = 0  # dictionary['UNK']
#                unk_count += 1
#            data.append(index) #storing index of dictionary
        try:
            index = dictionary[word]
        except KeyError:
            index = 0
            unk_count += 1
        #before assignment, count[0][1] = -1
        #after assigment, count[0][1] = 418391
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        return data, count, dictionary, reversed_dictionary
        # Step 3: Function to generate a training batch for the skip-gram model.
    def generate_batch(self, data):
        global data_index
        num_skips = self.config.num_skips
        skip_window = self.config.skip_window
        batch_size = self.config.batch_size
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        print(span)
        print(skip_window)
        print('Length data ', len(data))
        buffer = collections.deque(maxlen=span)
        if data_index + span > len(data):
            data_index = 0
        print('Data index', data_index)
        buffer.extend(data[data_index:data_index + span]) #copy data to buffer
        data_index += span
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window] #store the center word
                labels[i * num_skips + j, 0] = buffer[target]
        # for i in range(batch_size):
        #     batch[i] = buffer[skip_window]
        #     for j in range(span):
        #       if j != skip_window:
        #         labels[j] = buffer[j]
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index + len(data) - span) % len(data)
        return batch, labels

data_index = 0
def test_w2v():
    with tf.Graph().as_default():
        init = tf.global_variables_initializer()
        model = w2v_model()
        num_steps = 1000
        vocabulary = model.read_data()
        data,count, dictionary, reverse_dictionary = model.build_dataset(vocabulary)
        del vocabulary #reduce memory
        print('Most common words', count[:5])
        print('Sample data',data[:10], [reverse_dictionary[i] for i in data[:10]])
        with tf.Session() as sess:
            sess.run(init)
            print("Initialized!")
            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = model.generate_batch(data)


#    dictionary["history"]
#    batch, labels = w2v.generate_batch(data)
    # The data list stores index of word through whole corpus
    # for i in data:
    #     if i != 0:
    #         print(i, reversed_dictionary[i])
#    for i in range(w2v.config.batch_size):
#        print(reversed_dictionary[batch[i]],
#            '->', labels[i, 0], reversed_dictionary[labels[i, 0]])
        # print(batch[i])
data_index = 0
if __name__ == "__main__":
    test_w2v()
