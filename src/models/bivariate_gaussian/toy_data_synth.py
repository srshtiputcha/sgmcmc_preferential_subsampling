#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: toy_data_synth.py \
        --path-to-data=<data_path> 
"""

import numpy as np
import pandas as pd
import os
from docopt import docopt
from scipy import stats as ss

if __name__ == "__main__":
    arguments = docopt(__doc__)
    
    #extract file path for data
    path=arguments["--path-to-data"]
    os.chdir(path)
    #set seed
    np.random.seed(100)
    
    #set model parameters
    print("Generating data")
    N= 10**3
    theta_true = np.array([0., 1.]) #data mean
    sigma_x = np.array([[ 1*10**5 , 6*10**4], [6*10**4,  2*10**5]]) #data covariance matrix
    sigma_x_inv = np.linalg.inv(sigma_x) #data precision 
    
    #generate data
    data = ss.multivariate_normal.rvs(mean= theta_true, cov=sigma_x, size=N)
    
    #save csv
    print("Coverting to CSV")
    dat_df = pd.DataFrame(data)
    dat_df.columns=['x1', 'x2']
    save_path = './toy_bvg_synth.csv'
    dat_df.to_csv(save_path, ",", index=False)
    print("Saved to file")
    
