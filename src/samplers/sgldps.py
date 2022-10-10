#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:03:12 2021

@author: putchas
"""

#!/usr/bin/env python3
"""

SGLD-NUS sampler 

"""
import jax.numpy as jnp
from jax import random
import numpy as np
import time

def ps_preliminaries(theta_hat, gradf_i_batch, x, y):
    
    if y.all() == None:
        f_i_grad_list = jnp.array(gradf_i_batch(jnp.squeeze(theta_hat), x))   #evaluate gradient at posterior mode for all datapoints 
    else:
        f_i_grad_list = jnp.array(gradf_i_batch(jnp.squeeze(theta_hat), x, y))   #evaluate gradient at posterior mode for all datapoints 

    return f_i_grad_list



def exact_probs(theta, gradf_i_batch, x, y):
    
    """
    Function to evaluate the exact non-uniform subsampling probabilities
    
    Inputs:
    theta - current theta draw (array)
    gradf_i_batch - function  to evaluate batched f_i gradients (function)
    x - covariates (array)
    y - responses / array of None if not in model (array)
    
    Outputs:
    probs - probabilities for each datapoint (array)
    """
    
    #container for the probabilities
    N = x.shape[0]
    probs = np.zeros(N)
    
    if y.all() == None:
        grad_block = gradf_i_batch(jnp.squeeze(theta), x)
    else:
        grad_block = gradf_i_batch(jnp.squeeze(theta), x,y)
    
    for i in range(N):
        #iterate over datapoints
        term1 = grad_block[i]
        p = jnp.linalg.norm(term1)
        probs[i] = np.array(p)

    #normalise to get probabilities to sum to 1
    probs /= np.sum(probs)
        
    return probs       

def approx_probs(theta_hat, f_i_grad_list):
    
    """
    Function to evaluate the approximate non-uniform subsampling probabilities
    
    Inputs:
    theta_hat - posterior mode (array)
    f_i_grad_list - all f_i gradients at mode (array)
    
    Outputs:
    probs - probabilities for each datapoint (array)
    """
    
    #container for the probabilities
    N = f_i_grad_list.shape[0]
    probs = np.zeros(N)
    
    for i in range(N):
        #iterate over datapoints
        grad_i = f_i_grad_list[i]
        p = jnp.linalg.norm(grad_i)
        probs[i] = np.array(p)

    #normalise to get probabilities to sum to 1
    probs /= np.sum(probs)
    
    return probs       

def sgldps_grad(key, theta, gradf_0, gradf_i_batch, probs, x, y, n):
    
    """
    Function to evaluate SGLD-NUS stochastic gradient
    
    Inputs:    
    key - PRNG 
    theta - parameters (array)
    gradf_0 - gradient of negative log prior (function)
    gradf_i_batch - batched f_i gradients (function)
    probs - probabilities for each datapoint (array)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    n - subsample size (integer)
    
    Outputs:
    param_grad - stochastic gradient estimate at theta
    """
       
    N = x.shape[0]
    dim = theta.shape[0]
    key, subkey = random.split(key)
    idx_batch = random.choice(subkey, N, shape=(n,), p=probs)
    probs_sub = jnp.tile(probs[np.array(idx_batch)], reps=(dim, 1)).transpose()        
    #calculate stochastic gradient
    if y.all() == None:
        term1 =  gradf_i_batch(theta, x[idx_batch,:])/probs_sub
    else:
        term1 = gradf_i_batch(theta, x[idx_batch,:], y[np.array(idx_batch)])/probs_sub 
        
    param_grad= jnp.mean(term1, axis=0)  + gradf_0(theta)
        
    return param_grad


def sgldps_kernel(key, theta, gradf_0, gradf_i_batch, step, probs, x, y, n):
    
    """
    Function to update SGLD-NUS kernel 
    
    Inputs:    
    key - PRNG 
    theta - parameters (array)
    gradf_0 - gradient of negative log prior  (function)
    gradf_i_batch - batched f_i gradients (function)
    step - step-size tuned for sampler (float)
    probs - probabilities for each datapoint (array)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    n - subsample size (integer)
    
    Outputs:
    theta - updated parameter values 
    param_grad - stochastic gradient 
    """
       
    N = x.shape[0]
    dim = theta.shape[0]
    subkey1, subkey2 = random.split(key)
    idx_batch = random.choice(subkey1, N, shape=(n,), p=probs)
    probs_sub = jnp.tile(probs[np.array(idx_batch)], reps=(dim, 1)).transpose()        
    
    #calculate stochastic gradient
    if y.all() == None:
        term1 =  gradf_i_batch(theta, x[idx_batch,:])/probs_sub
    else:
        term1 = gradf_i_batch(theta, x[idx_batch,:], y[np.array(idx_batch)])/probs_sub 
        
    param_grad= jnp.mean(term1, axis=0)  + gradf_0(theta) 
           
    #update theta
    theta = theta - (step/2)*param_grad + jnp.sqrt(step)*random.multivariate_normal(key = subkey2, mean = jnp.zeros(dim), cov = jnp.eye(dim))

    return theta, param_grad


def sgldps_sampler(key, gradf_0, gradf_i_batch, Niter, step, theta_0, theta_hat, x, y, n,  prob_type):
    
    """
    Function to run SGLD sampler
    
    Inputs:    
    key - PRNG 
    gradf_0 - gradient of negative log prior (function)
    gradf_i_batch - batched f_i gradients (function)
    Niter - number of samples (integer)
    step - step-size tuned for sampler (float)
    theta_0 - initial parameters (array)
    theta_hat - posterior mode (array)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    n - subsample size (integer)
    prob_type - specifies 'exact' or 'approx' NUS probabilities (string)

    Outputs:
    samples - parameter values (array)
    grads - stochastic gradients (array)
    """
    
    N = x.shape[0]
    dim = theta_0.shape[0]
    samples = np.zeros((Niter+1, dim))
    grads = np.zeros((Niter, dim))
    run_time = np.zeros(Niter)
    
    x = jnp.array(x)
    if y.all() != None:
        y = jnp.array(y)
        
    f_i_grad_list = ps_preliminaries(theta_hat, gradf_i_batch, x, y)
    samples[0,:]=theta_0
    theta = theta_0 
        
    if prob_type == 'exact':
        start_time = time.time()
        for i in range(Niter):
            key, subkey = random.split(key)
            probs = exact_probs(theta, gradf_i_batch, x, y)
            theta, param_grad = sgldps_kernel(subkey, theta, gradf_0, gradf_i_batch, step, probs, x, y, n)
            samples[i+1,:] = theta
            grads[i] = param_grad
            iter_time = time.time()
            run_time[i] = iter_time - start_time

    else:
        probs = approx_probs(theta_hat, f_i_grad_list)
        start_time = time.time()
        for i in range(Niter):
            key, subkey = random.split(key)
            theta, param_grad = sgldps_kernel(subkey, theta, gradf_0, gradf_i_batch, step, probs, x, y, n)
            samples[i+1,:] = theta
            grads[i] = param_grad
            iter_time = time.time()
            run_time[i] = iter_time - start_time

    return samples, grads, run_time



        
        
    
        
    
    
