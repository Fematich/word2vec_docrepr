#!/usr/bin/env python
"""
@author:    Matthias Feys (matthiasfeys@gmail.com), IBCN (Ghent University)
@date:      Wed Oct 30 12:38:52 2013
"""
"""
Module for transforming a text document into a bag of wordembeddings
"""
import logging,sys,os
import dataprocessing
from nltk.corpus import stopwords
from dataprocessing.muc import MUCmessages
from gensim.models.word2vec import Word2Vec

logger = logging.getLogger("bagmodel")

class BagEmbeddings():
    '''
    class around gensim Word2Vec model, to transform complete document into a Word2Vec bag of words
    '''
    def __init__(self,modelpath='/home/mfeys/work/data/word2vec',model='vectors600.bin',stopwords=None):
        self.model = Word2Vec.load_word2vec_format(os.path.join(modelpath,model), binary=True)  # C binary format
        if stopwords==None:
            self.stopwords=stopwords.words('english')
        else:
            self.stopwords=stopwords
    
    def transformdoc(doc):
        texts = [word for word in document.lower().split() if word not in stoplist] for document in documents]

    def transformcorpus(corpus):
        for doc in corpus:
            yield self.transformdoc(doc)


if __name__ == '__main__':
    stopwords.words('english')
    tel=0
    msgs=MUCmessages()
    res,ndocs=msgs.finddocs('guatemala',10)
    print ndocs
    print len(res)
    for msg in res:
        print msg['content'].lower()
        test
