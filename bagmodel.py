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

from dataprocessing.muc import MUCmessages

logger = logging.getLogger("bagmodel")
tel=0
msgs=MUCmessages()
res,ndocs=msgs.finddocs('guatemala',10)
print ndocs
print len(res)
for msg in res:
    print msg['content'].lower()
