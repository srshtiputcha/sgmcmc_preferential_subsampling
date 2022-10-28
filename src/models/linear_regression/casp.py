#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage: casp.py \
        --path-to-data=<data_path> 
"""
import os
import pandas as pd
from docopt import docopt
from sklearn import preprocessing
import numpy as np
  
if __name__ == "__main__":
    arguments = docopt(__doc__)
    
    #extract file paths 
    path=arguments["--path-to-data"]
    os.chdir(path)
    file = 'CASP.csv'
  
    #preprocess raw data
    casp_df = pd.read_csv(file).dropna()
    
    standard_scaler = preprocessing.StandardScaler() #standardise all variables
    casp_df.iloc[:,:] = standard_scaler.fit_transform(casp_df.values[:,:])
   
    #save proprecessed dataset                                        
    casp_df.to_csv("casp_scaled.csv", index=False)

