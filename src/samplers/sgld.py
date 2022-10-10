#!/usr/bin/env python3
"""

SGLD sampler 

"""
import jax.numpy as jnp
from jax import jit, random
import numpy as np
import time
from tqdm import tqdm

def sgld_grad(key, theta, gradf_0, gradf_i_batch, x, y, n, replacement):
    
    """
    Function to evaluate SGLD stochastic gradient
    
    Inputs:    
    key - PRNG 
    theta - parameters (array)
    gradf_0 - gradient of negative log prior (function)
    gradf_i_batch - batched f_i gradients (function)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    n - subsample size (integer)
    replacement - boolean (True or False) for sampling with/with-out replacement
    
    Outputs:
    param_grad - stochastic gradient estimate at theta
    """
       
    N = x.shape[0]
    key, subkey = random.split(key)
    idx_batch = random.choice(subkey, N, shape=(n,), replace = replacement)
    #calculate stochastic gradient
    if y.all() == None:
        param_grad =  N*jnp.mean(gradf_i_batch(theta, x[idx_batch,:]),axis=0) + gradf_0(theta)
        
    else:
        param_grad = N*jnp.mean(gradf_i_batch(theta, x[idx_batch,:], y[np.array(idx_batch)]), axis=0) + gradf_0(theta)
        
    return param_grad


def sgld_kernel(key, theta, gradf_0, gradf_i_batch, step, x, y, n, replacement):
    
    """
    Function to update SGLD kernel 
    
    Inputs:    
    key - PRNG 
    theta - parameters (array)
    gradf_0 - gradient of negative log prior  (function)
    gradf_i_batch - batched f_i gradients (function)
    step - step-size tuned for sampler (float)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    n - subsample size (integer)
    replacement - boolean (True or False) for sampling with/with-out replacement
    
    Outputs:
    theta - updated parameter values 
    param_grad - stochastic gradient 
    """
       
    N = x.shape[0]
    subkey1, subkey2 = random.split(key)
    dim = theta.shape[0]
    idx_batch = random.choice(subkey1, N, shape=(n,), replace=replacement)
    
    #calculate stochastic gradient
    if y.all() == None:
        param_grad =  N*jnp.mean(gradf_i_batch(theta, x[idx_batch,:]),axis=0) + gradf_0(theta)
    else:
        param_grad = N*jnp.mean(gradf_i_batch(theta, x[idx_batch,:], y[np.array(idx_batch)]), axis=0) + gradf_0(theta)
        
    #update theta
    
    theta = theta - (step/2)*param_grad + jnp.sqrt(step)*random.multivariate_normal(key = subkey2, mean = jnp.zeros(dim), cov = jnp.eye(dim))
    
    return theta, param_grad


def sgld_sampler(key, gradf_0, gradf_i_batch, Niter, step, theta_0, x, y, n, replacement):
    
    """
    Function to run SGLD sampler
    
    Inputs:    
    key - PRNG 
    gradf_0 - gradient of negative log prior (function)
    gradf_i_batch - batched f_i gradients (function)
    Niter - number of samples (integer)
    step - step-size tuned for sampler (float)
    theta_0 - initial parameters (array)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    n - subsample size (integer)
    replacement - boolean (True or False) for sampling with/with-out replacement
    
    Outputs:
    samples - parameter values (array)
    grads - stochastic gradients (array)
    """
    
    N = np.shape(x)[0]
    dim = theta_0.shape[0]
    samples = np.zeros((Niter+1, dim))
    grads = np.zeros((Niter, dim))
    run_time = np.zeros(Niter)
    
    x = jnp.array(x)
    if y.any() != None:
        y = jnp.array(y)
        
    samples[0,:]=theta_0
    theta = theta_0 
    start_time = time.time()
    for i in range(Niter):
        key, subkey = random.split(key)
        theta, param_grad = sgld_kernel(subkey, theta, gradf_0, gradf_i_batch, step, x, y, n, replacement)
        samples[i+1] = theta
        grads[i] = param_grad
        iter_time = time.time()
        run_time[i] = iter_time - start_time

    return samples, grads, run_time



        
        
    
        
    
    
