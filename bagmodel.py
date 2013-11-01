#!/usr/bin/env python
"""
@author:    Matthias Feys (matthiasfeys@gmail.com), IBCN (Ghent University)
@date:      Wed Oct 30 12:38:52 2013
"""
"""
Module for transforming a text document into a bag of wordembeddings
"""
import numpy as np
import logging,sys,os
import cPickle as pickle
import dataprocessing
from nltk.corpus import stopwords as NLTKStopwords
from nltk import word_tokenize, wordpunct_tokenize
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
            self.stopwords=NLTKStopwords.words('english')
        else:
            self.stopwords=stopwords
    
    def transformdoc(self,doc):
        document=np.zeros(self.model.layer1_size)
        for word in [word_tokenize(sentence) for sentence in wordpunct_tokenize(doc.lower())]:
            if word[0] not in self.stopwords:
                try:
                    document+=self.model[word[0]]
                except Exception:
                    pass
        return document

    def streamtransformcorpus(self,corpus):
        for doc in corpus:
            yield self.transformdoc(doc)


if __name__ == '__main__':
    msgs=MUCmessages()
    model=BagEmbeddings()
#    res,ndocs=msgs.finddocs('guatemala',10)
#    for msg in res:
#        print model.transformdoc(msg['content'])
#    corpus=np.array([model.transformdoc(doc[1]['content'].lower()) for doc in msgs])
#    pickle.dump(corpus,open('../data/corpus.pck','w'))
#    logger.info('done')

    #todo: add types to whoosh index
    