# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:13:29 2017

@author: hungtran
#Goal: Create word vectors from corpus using gensim
"""
from __future__ import absolute_import, division, print_function
#for word encoding
import codecs
#regex
import glob
#concurrency
import multiprocessing
#dealing with operation system, like reading a file
import os
#pretty printing, human readable
import pprint
#regular expression
import re
#natural langugage toolkit
import nltk
#word 2 vec
import gensim.models.word2vec as w2v
from gensim import models
import time
class params:
    def __init__(self):
        self.data_path = os.getcwd() + "\\data\\*.txt"
        #Dimensionality of the resulting word vectors
        self.num_features = 300
        #Minimum word count threshold
        #smallest set of words that we want to recognize, when we convert to a vector
        self.min_word_count = 3
        #Number of threads to run in parallel
        self.num_workers = multiprocessing.cpu_count()
        #Context window length
        self.context_size = 7
        #Downsmaple setting for frequent words
        #how frequent we want to see a word, the more the ward appear the less we want to use it to create a vector
        self.downsampling = 1e-3
        #Seed for the RNG, to mae the results reproducible
        #random number generator, use to pick what part of corpus we want to create a vector
        #deterministic, good for debugging.
        self.seed = 1

class word2vec_model:
    def __init__(self,params):
        self.params = params
        self.sentences = self.get_sentences()
        # self.wordlist = self.sentence_to_wordlist(self.sentences)
        self.word2vec = self.word2vec_op()

    def loadData(self,path):
        text_file_names = self.readFiles(path)
        #Combine the books into one string
        corpus_raw = u""
        for file in text_file_names:
            pprint.pprint("Reading %s ..."%(file))
            with codecs.open(file,"r", "utf-8") as text_file:
                corpus_raw += text_file.read()
                pprint.pprint(" Corpus is now %i characters long" %len(corpus_raw))
        return corpus_raw
    def readFiles(self,path):
        return sorted(glob.glob(path))
    def get_raw_sentences(self):
        nltk.download('punkt') #pretrained tokenizer
        nltk.download('stopwords') #words like the, a, an, of, and
        #split corpus into sentences
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        corpus_raw = self.loadData(self.params.data_path)
        raw_sentences = tokenizer.tokenize(corpus_raw)
        return raw_sentences
    def sentence_to_words(self,sentence):
        clean = re.sub("[^a-zA-Z]"," ",sentence)
        words = clean.split()
        return words
    def get_sentences(self):
        raw_sentences = self.get_raw_sentences()
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(self.sentence_to_words(raw_sentence))
        token_count = sum([len(sentence) for sentence in sentences])
        pprint.pprint("Number of sentences is %i" %len(sentences))
        pprint.pprint("Number of token is %i" %token_count)
        return sentences
    def word2vec_op(self):
        #Build model
        word2vec = w2v.Word2Vec(
                sg=1,
                seed=self.params.seed,
                workers=1,
                size=self.params.num_features,
                min_count=self.params.min_word_count,
                window=self.params.context_size,
                sample=self.params.downsampling
                )
        return word2vec
    def train_op(self):
        self.word2vec.build_vocab(self.sentences)
        pprint.pprint("Vocabulary size: %i"%len(self.word2vec.wv.vocab))
        #Start training
        self.word2vec.train(self.sentences,total_examples=self.word2vec.corpus_count,epochs=self.word2vec.iter)
    def save_to_file(self):
        if not(os.path.exists('trained')):
            os.makedirs('trained')
        self.word2vec.save(os.path.join('trained','saved_word2vec.w2v')) 
def test_word2vec():
    _params = params()
    start = time.time()
    model = word2vec_model(_params)
    model.train_op()
    model.save_to_file()
    print ("Time passed {:.2f} ".format(float(time.time() - start)))
if __name__ == "__main__":
    test_word2vec()