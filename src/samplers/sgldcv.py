#!/usr/bin/env python3
"""

SGLD-CV sampler 

"""
import jax.numpy as jnp
from jax import random
import numpy as np
import time

def cv_preliminaries(theta_hat, gradf_0, gradf_i_batch, x, y):
    
    """
    Function to compute regularly used quantities
    
    Inputs:
    theta_hat - mode (array)
    gradf_0 - gradient of f_0 (function)
    gradf_i_batch - batched f_i gradients (functions)
    x - covariates (array)
    y - response / array of None if not in model (array)
    
    Outputs:
    gradf_0_hat - gradient of f_0 at the mode (array)
    grad_full_hat - sum of gradients of f_i's at the mode (array)
    f_i_grad_list - gradients of f_i's at the mode (array)
    """
    
    #evaluate gradient of f_0 at the posterior mode
    gradf_0_hat = gradf_0(theta_hat)

    #evaluate gradient at posterior mode for all datapoints (array reused when minibatching)
    if y.all() == None:
        f_i_grad_list = jnp.array(gradf_i_batch(jnp.squeeze(theta_hat), x))
    else:
        f_i_grad_list = jnp.array(gradf_i_batch(jnp.squeeze(theta_hat), x, y))
                          
    #sum over all datapoints
    grad_full_hat = jnp.sum(f_i_grad_list, axis = 0)
                          
    return gradf_0_hat, grad_full_hat, f_i_grad_list

def sgld_cv_grad(key, theta, theta_hat, gradf_0, gradf_i_batch, grad_full_hat, f_i_grad_list, x, y, n, replacement):
    
    """
    Function to evaluate SGLD-CV gradient  
    
    Inputs:    
    key - PRNG 
    theta - parameters (array)
    theta_hat - posterior mode (array)
    gradf_0 - gradient of f_0 (function)
    gradf_i_batch - batched f_i gradients (function)
    grad_full_hat - sum of all f_i gradients at mode (array)
    f_i_grad_list - f_i gradients for all points (array)
    x - observations (array)
    y - response variables / array of None if not in model (array)
    n - subsample size (integer)
    replacement - boolean (True or False) for sampling with/with-out replacement
    
    Outputs:
    param_grad - stochastic gradient estimate
    """
       
    N = x.shape[0]
    key, subkey = random.split(key)
    idx_batch = random.choice(subkey, N, shape=(n,), replace = replacement)
    
    
    #calculate stochastic gradient
    if y.all() == None:
        term1 =  gradf_i_batch(theta, x[idx_batch,:])
    else:
        term1 = gradf_i_batch(theta, x[idx_batch,:], y[np.array(idx_batch)])
    term2 = f_i_grad_list[idx_batch]
    
    param_grad = gradf_0(theta) + grad_full_hat + (N/n)*jnp.sum(term1-term2, axis=0)
    return param_grad


def sgld_cv_kernel(key, theta, theta_hat, gradf_0, gradf_i_batch, grad_full_hat, f_i_grad_list, step, x, y, n, replacement):
    
    """
    Function to update SGLD-CV kernel 
    
    Inputs:    
    key - PRNG 
    theta - parameters (array)
    theta_hat - posterior mode (array)
    gradf_0 - gradient of f_0 (function)
    gradf_i_batch - batched f_i gradients (function)
    grad_full_hat - sum of all f_i gradients at mode (array)
    f_i_grad_list - f_i gradients for all points (array)
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
        term1 =  gradf_i_batch(theta, x[idx_batch,:])
    else:
        term1 = gradf_i_batch(theta, x[idx_batch,:], y[np.array(idx_batch)])
    term2 = f_i_grad_list[idx_batch]
    
    param_grad = gradf_0(theta) + grad_full_hat + (N/n)*jnp.sum(term1-term2, axis=0)
    
    #update theta
    theta = theta - (step/2)*param_grad + jnp.sqrt(step)*random.multivariate_normal(key = subkey2, mean = jnp.zeros(dim), cov = jnp.eye(dim))
    return theta, param_grad 


def sgld_cv_sampler(key, gradf_0, gradf_i_batch, Niter, step, theta_0, theta_hat, x, y, n, replacement):
    
    """
    Function to run SGLD-CV sampler
    
    Inputs:
    key - PRNG 
    theta_0 - initial parameters (array)
    theta_hat - mode (array)
    gradf_0 - gradient of negative log prior (function)
    gradf_i_batch - batched f_i gradients (function)
    Niter - number of samples (integer)
    step - step-size tuned for sampler (float)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    n - subsample size (integer)
    replacement - boolean (True or False) for sampling with/with-out replacement
    
    Outputs:
    samples - parameter values (array)
    grads - stochastic gradients (array)
    run_time - runtime of sampler 
    """
    
    N = np.shape(x)[0]
    dim = theta_0.shape[0]
    samples = np.zeros((Niter+1, dim))
    grads = np.zeros((Niter, dim))
    run_time = np.zeros(Niter)
    
    x = jnp.array(x)
    if y.all() != None:
        y = jnp.array(y)
        
    gradf_0_hat, grad_full_hat, f_i_grad_list = cv_preliminaries(theta_hat, gradf_0, gradf_i_batch, x, y)
        
    samples[0,:]=theta_0
    theta = theta_0
    start_time = time.time()
    for i in range(Niter):
        key, subkey = random.split(key)
        theta, param_grad = sgld_cv_kernel(subkey, theta, theta_hat, gradf_0, gradf_i_batch, 
                                           grad_full_hat, f_i_grad_list, step, x, y, n, replacement)
        samples[i+1] = theta
        grads[i] = param_grad
        iter_time = time.time()
        run_time[i] = iter_time - start_time

    return samples, grads, run_time


def asgld_cv_sampler(key, gradf_0, gradf_i_batch, Niter, step, theta_0, 
                     theta_hat, x, y, cons, replacement):
    
    """
    Function to run adaptive SGLD-CV sampler
    
    Inputs:
    key - PRNG 
    theta_0 - initial parameters (array)
    theta_hat - mode (array)
    gradf_0 - gradient of negative log prior (function)
    gradf_i_batch - batched f_i gradients (function)
    Niter - number of samples (integer)
    step - step-size tuned for sampler (float)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    cons - multiplicative constant to scale ||\theta - \theta_hat ||^2 to select n (scalar)
    replacement - boolean (True or False) for sampling with/with-out replacement
    
    Outputs:
    samples - parameter values (array)
    grads - stochastic gradients (array)
    run_time - runtime of sampler (array)
    n_vec - batch sizes chosen each iteration (array)
    """
    
    N = np.shape(x)[0]
    dim = theta_0.shape[0]
    samples = np.zeros((Niter+1, dim))
    grads = np.zeros((Niter, dim))
    run_time = np.zeros(Niter)
    n_vec = np.zeros(Niter)
    
    x = jnp.array(x)
    if y.all() != None:
        y = jnp.array(y)
        
    gradf_0_hat, grad_full_hat, f_i_grad_list = cv_preliminaries(theta_hat, gradf_0, gradf_i_batch, x, y)
    
    samples[0,:]=theta_0
    theta = theta_0
    start_time = time.time()
    for i in range(Niter):
        key, subkey = random.split(key)

        #squared l2 distance between theta and mode 
        temp = theta - theta_hat
        dist2 = np.dot(temp.T, temp)

        #adaptively select batch size
        n = max(1, np.ceil(cons*dist2).astype(int))
        if n > N:
            n = N 
        n_vec[i] = n
        theta, param_grad = sgld_cv_kernel(subkey, theta, theta_hat, gradf_0, gradf_i_batch, 
                                           grad_full_hat, f_i_grad_list, step, x, y, n, replacement)
        samples[i+1] = theta
        grads[i] = param_grad
        iter_time = time.time()
        run_time[i] = iter_time - start_time

    return samples, grads, run_time, n_vec
        
