from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import nltk
from nltk.corpus import stopwords
import time
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

special_marks = [".",",",";","!","'",":"]
class config:
    def __init__(self):
        self.stopwords = stopwords.words("english")
        for mark in special_marks:
            self.stopwords.append(mark)
        self.vocabulary_size = 200
        self.filename = os.getcwd() + "\\data\\LoR_new"
        self.batch_size = 12
        self.embedding_size = 128  # Dimension of the embedding vector.
        self.skip_window = 3       # How many words to consider left and right.
        self.num_skips = 2        # How many times to reuse an input to generate a label.
class w2v_model():
    def __init__(self):
        self.config = config()
    # Read the data into a list of strings.
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
        print ("Read data took {:.2f} second".format(time.time()-start))
        print ("Size of vocabulary", len(processed_word_list), "including recurrence")
        #print ("Stop words contains: ", self.config.stopwords)
        return processed_word_list
    def build_dataset(self,words): #words = processed word list
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
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index) #storing index of dictionary
        #before assignment, count[0][1] = -1
        #after assigment, count[0][1] = 418391
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        return data, count, dictionary, reversed_dictionary
        # Step 3: Function to generate a training batch for the skip-gram model.
    def generate_batch(self,data):
        global data_index
        num_skips = self.config.num_skips
        skip_window = self.config.skip_window
        batch_size = self.config.batch_size
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        if data_index + span > len(data):
            data_index = 0
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

def test_w2v():
    w2v = w2v_model()
    vocabulary = w2v.read_data()
    data, count, dictionary, reversed_dictionary = w2v.build_dataset(vocabulary)
    del vocabulary #reduce memory
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reversed_dictionary[i] for i in data[:10]])
    dictionary["history"]
    batch, labels = w2v.generate_batch(data)
    # The data list stores index of word through whole corpus
    # for i in data:
    #     if i != 0:
    #         print(i, reversed_dictionary[i])
    for i in range(w2v.config.batch_size):
        print(reversed_dictionary[batch[i]],
            '->', labels[i, 0], reversed_dictionary[labels[i, 0]])
        # print(batch[i])
data_index = 0    
if __name__ == "__main__":
    test_w2v()
