# -*- coding: utf-8 -*-
"""

@author: LMC_ZC
"""

import sys
from scipy import stats


class Logger(object):
    def __init__(self, fileN="Default.logs"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()
        
        
def conditional_samples(e):
    anchor_e = e[0]
    gid = [0]
    
    for k in range(1, e.shape[0]):
        if stats.pearsonr(e[0], e[k])[0] > 0:
            gid += [0]
        else:
            gid += [1]
            
    return gid