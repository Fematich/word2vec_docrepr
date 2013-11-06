#!/usr/bin/env python
"""
@author:    Matthias Feys (matthiasfeys@gmail.com), IBCN (Ghent University)
@date:      Fri Nov  1 16:00:02 2013
"""
'''
Given a set of training=documents and a set of test-documents as ..
it performs SVM-classification from SK-learn and returns the performance
'''

from sklearn import svm
from dataprocessing.muc import MUCkeys
import cPickle as pickle

types=['BOMBING','ATTACK','KIDNAPPING','ARSON']

if __name__ == '__main__':
    #get documentrepresentations
    corpus=pickle.load(open('../data/corpus.pck','r'))
    X_train=corpus[:1400]
    X_test=corpus[1400:]
    #get labels
    keys=MUCkeys()
    for muctype in types:
        print '==============='
        print muctype
        print '---------------'
        labels=keys.GetAllTypePresences(muctype)
        Y_train=labels[:1400]
        Y_test=labels[1400:]
        #train SVM
    #    clf = svm.SVC(kernel='poly')
        clf = svm.LinearSVC()
        clf.fit(X_train, Y_train)
        
        #get performace on test-set
        tp=0;tn=0;fp=0;fn=0
        
        for tel in xrange(len(X_test)):
            prediction=clf.predict([X_test[tel]])[0]
            #print  prediction,Y_test[tel]
            if  prediction==Y_test[tel]:
                if prediction==1:
                    tp+=1
                else:
                    tn+=1
            else:
                if prediction==1:
                    fp+=1
                else:
                    fn+=1
        print 'tp:%d\ttn:%d\tfp:%d\tfn:%d'%(tp,tn,fp,fn)    
        precision=float(tp)/(tp+fp)
        recall=float(tp)/(tp+fn)
        print 'PRECISION=%f'%precision
        print 'RECALL=%f'%recall
        print 'F1=%f'%(float(2*precision*recall)/(precision+recall))
