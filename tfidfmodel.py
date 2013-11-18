#!/usr/bin/env python
"""
@author:    Matthias Feys (matthiasfeys@gmail.com), IBCN (Ghent University)
@date:      Mon Nov 18 11:02:28 2013
"""

from gensim import corpora, matutils
from gensim.corpora import TextCorpus, Dictionary
import logging

from nltk.corpus import stopwords as NLTKStopwords
from nltk import word_tokenize, wordpunct_tokenize
from dataprocessing.muc import MUCmessages
import cPickle as pickle


logger = logging.getLogger("tfidfmodel")

class CorpusMUC(corpora.TextCorpus):
    def __init__(self):
        super(TextCorpus, self).__init__()
        self.stopwords=NLTKStopwords.words('english')  
        self.stopwords.extend(['``',',','(',')','.'])
        self.msgs=MUCmessages()
        self.dictionary = Dictionary()
        self.dictionary.add_documents(self.get_texts())
        
    def get_texts(self):
        """
        Parse documents from the .cor file provided in the constructor. Lowercase
        each document and ignore some stopwords.

        .cor format: one document per line, words separated by whitespace.
        """
        for doc in self.msgs:
            document=[word for word in [word_tokenize(sentence) for sentence in wordpunct_tokenize(doc[1]['content'].lower())]]
            yield [str(word[0]) for word in document if str(word[0]) not in self.stopwords]

    def __len__(self):
        """Define this so we can use `len(corpus)`"""
        if 'length' not in self.__dict__:
            logger.info("caching corpus size (calculating number of documents)")
            self.length = sum(1 for doc in self.get_texts())
        return self.length
        

if __name__ == '__main__':  
    MUCCorpus=CorpusMUC()
    mtrx=matutils.corpus2csc(MUCCorpus,num_terms=17204,num_docs=1800,num_nnz=230448,printprogress=0)
    mtrx.transpose()
    print 'nnz:',mtrx.nnz
    print 'shape:',mtrx.shape
    pickle.dump(mtrx,open('../data/TFCorpus.pck','w'))
    logger.info('done')
    