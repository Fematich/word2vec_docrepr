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
import os
from mongostore.mongostore import MongoStore

types=['BOMBING','ATTACK','KIDNAPPING','ARSON']

@MongoStore
def SVMClassifier(corpusname):
    #get documentrepresentations
    corpus=pickle.load(open(os.path.join('../data',corpusname),'r'))
    X_train=corpus[:1400]
    X_test=corpus[1400:]
    #get labels
    keys=MUCkeys()
    ret={}
    for muctype in types:
        print '==============='
        print muctype
        print '---------------'
        labels=keys.GetAllTypePresences(muctype)
        Y_train=labels[:1400]
        Y_test=labels[1400:]
        #train SVM
#        clf = svm.SVC(kernel='poly')
        clf = svm.LinearSVC()
        clf.fit(X_train, Y_train)
        
        #get performace on test-set
        tp=0;tn=0;fp=0;fn=0
        
        for tel in xrange(len(Y_test)):
            prediction=clf.predict(X_test[tel])[0]
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
        try:
            ret[muctype]={'tp':tp,'tn':tn,'fp':fp,'fn':fn}
            print 'tp:%d\ttn:%d\tfp:%d\tfn:%d'%(tp,tn,fp,fn)    
            precision=float(tp)/(tp+fp)
            ret[muctype]['precision']=precision
            recall=float(tp)/(tp+fn)
            ret[muctype]['recall']=recall
            print 'PRECISION=%f'%precision
            print 'RECALL=%f'%recall
            F1=float(2*precision*recall)/(precision+recall)
            print 'F1=%f'%(F1)
            ret[muctype]['F1']=F1
        except Exception:
            continue
    return ret
if __name__ == '__main__':
    SVMClassifier(corpusname='corpus.pck')
