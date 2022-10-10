#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: toy_data_synth_imbalanced.py \
        --path-to-data=<data_path> 
"""

import numpy as np
import pandas as pd
import os 
from docopt import docopt
from sklearn.datasets import make_classification #for generating synthetic data
from sklearn.model_selection import train_test_split #to split the data

if __name__ == "__main__":
    arguments = docopt(__doc__)
    
    #extract file path for data
    path=arguments["--path-to-data"]
    os.chdir(path)
    
    #set seed
    np.random.seed(100)
    
    #set model parameters
    N= 10**3  #number of training data points
    N_test = int(N/2) #number of test data points
    dim = 4    # number of covariates
    
    #generate test and training data 
    print("Generating data")
    X_full, y_full = make_classification(
    n_samples=N + N_test,
    n_features=dim,
    n_classes=2,
    n_clusters_per_class=1,
    #introduces some random class allocation
    weights = [0.01, 0.99], #1% of data in {y = 0} and 99% of data in {y=1}
    n_informative=4, #number of informative variables
    n_redundant=0,
    n_repeated=0
    )
    
    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=1/3, random_state=42)
    
    print("Coverting to CSV")
    intercept_train = np.ones(N)
    df_train = pd.DataFrame(np.column_stack((y_train, intercept_train, X_train)))
    df_train.columns = ['y', 'x0', 'x1', 'x2', 'x3', 'x4']
    
    intercept_test = np.ones(N_test)
    df_test = pd.DataFrame(np.column_stack((y_test, intercept_test, X_test)))
    df_test.columns = ['y', 'x0', 'x1', 'x2', 'x3', 'x4']
    
    #save csv
    print("Saving traing dataset")
    save_path = './toy_lr_imbalance_train_synth.csv'
    df_train.to_csv(save_path, ",", index=False)
    print("Saving test datset" )
    save_path = './toy_lr_imbalance_test_synth.csv'
    df_test.to_csv(save_path, ",", index=False)
    
    print("Saved to file")
    
