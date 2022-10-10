#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage: covtype.py \
        --path-to-data=<data_path> 
"""
import re
import os
import pandas as pd
from docopt import docopt
from sklearn import preprocessing
import numpy as np

def writeFile(inputfile, outputfile, instances):
   
  wf = open(outputfile, 'w')
  values = [0.0] * (instances + 1)
  with open(inputfile) as rf:
    for line in rf:
      matches = re.findall('([-+]?[0-9]*\.?[0-9]+):([-+]?[0-9]*\.?[0-9]+)', line, re.DOTALL)
      values[0] = 0 if line[0] == '1' else 1
      for index, value in matches:
        values[int(index)] = float(value)
      wf.write(','.join(map(str, values)) + '\n')
      values = [0.0] * (instances + 1)
  wf.close()
  
if __name__ == "__main__":
    arguments = docopt(__doc__)
    
    #extract file paths 
    path=arguments["--path-to-data"]
    os.chdir(path)
    inputfile = 'covtype.libsvm.binary'
    outputfile= 'covtype_binary.csv'
  
    #convert raw file to CSV 
    writeFile(inputfile, outputfile, 54)
    
    #preprocess raw data
    covtype = pd.read_csv("covtype_binary.csv", header=None)
    np.random.seed(10)
    
    robust_scaler = preprocessing.RobustScaler() #standardize numerical variables
    covtype.iloc[:, 1:11] = robust_scaler.fit_transform(covtype.values[:, 1:11])

    #split into test-train data, preserving class balance
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.25, random_state=0)

    for train_index, test_index in sss.split(covtype, covtype[0]):
        cover_train = covtype.iloc[train_index].reset_index(drop=True)
        cover_test = covtype.iloc[test_index].reset_index(drop=True)
        
    #save test-train datasets 
    cover_train.to_csv("covtype_train.csv", index=False)
    cover_test.to_csv("covtype_test.csv", index=False)
    

