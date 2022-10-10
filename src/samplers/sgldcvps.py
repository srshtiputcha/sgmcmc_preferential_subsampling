#!/usr/bin/env python3
"""

SGLD-CV-PS sampler 

"""
import jax.numpy as jnp
from jax import random
import numpy as np
import time

def cvps_preliminaries(theta_hat, gradf_0, gradf_i_batch, post_var, x, y):
    
    """
    Function to compute regularly used quantities
    
    Inputs:
    theta_hat - mode (array)
    gradf_0 - gradient of f_0 (function)
    gradf_i_batch - batched f_i gradients (function)
    post_var - evaluate covariance matrix of Normal approximation (function)
    x - covariates (array)
    y - response / array of None if not in model (array)
    
    Outputs:
    cov_mat - covariate matrix of Normal approximation (matrix)
    gradf_0_hat - gradient of f_0 at the mode (array)
    grad_full_hat - sum of gradients of f_i's at the mode (array)
    f_i_grad_list - gradients of f_i's at the mode (array)
    f_i_hess_list - hessians of f_i's at the mode (array)
    """
    
    #evaluate gradient of f_0 at the posterior mode
    gradf_0_hat = gradf_0(theta_hat)

    if y.all() == None: #evaluate gradient, hessian at posterior mode for all datapoints 
        f_i_grad_list = jnp.array(gradf_i_batch(jnp.squeeze(theta_hat), x))   
        cov_mat, f_i_hess_list = post_var(theta_hat, x) 
    else:
        f_i_grad_list = jnp.array(gradf_i_batch(jnp.squeeze(theta_hat), x, y))   
        cov_mat, f_i_hess_list = post_var(theta_hat, x, y) 
                          
    #sum over all datapoints
    grad_full_hat = jnp.sum(f_i_grad_list, axis = 0)
    
    return cov_mat, gradf_0_hat, grad_full_hat, f_i_grad_list, f_i_hess_list


def exact_probs(theta, theta_hat, gradf_i_batch, f_i_grad_list, x, y):
    
    """
    Function to evaluate the exact non-uniform subsampling probabilities
    
    Inputs:
    theta - current theta draw (array)
    theta_hat - posterior mode (array)
    gradf_i_batch - function  to evaluate batched f_i gradients (function)
    f_i_grad_list - all f_i gradients at mode (array)
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
        term2 = f_i_grad_list[i]
        p = jnp.linalg.norm(term1 - term2)
        probs[i] = np.array(p)

    #normalise to get probabilities to sum to 1
    probs /= np.sum(probs)
        
    return probs       

def approx_probs(theta_hat, f_i_hess_list, cov_mat):
    
    """
    Function to evaluate the approximate non-uniform subsampling probabilities
    
    Inputs:
    theta_hat - posterior mode (array)
    f_i_hess_list - all f_i hessians at mode (array)
    cov_mat - estimated covariance matrix of normal approximation (matrix)
    
    Outputs:
    probs - probabilities for each datapoint (array)
    """
    
    #container for the probabilities
    N = f_i_hess_list.shape[0]
    probs = np.zeros(N)
    sigma_hat = cov_mat
    
    for i in range(N):
        #iterate over datapoints
        hess_mat_i = f_i_hess_list[i]
        term1 = jnp.matmul(hess_mat_i, sigma_hat)
        term2 = jnp.matmul(term1, hess_mat_i.transpose())
        p = jnp.sqrt(jnp.trace(term2))
        probs[i] = np.array(p)

    #normalise to get probabilities to sum to 1
    probs /= np.sum(probs)
    
    return probs       
    

def sgldcv_ps_grad(key, theta, theta_hat, gradf_0, gradf_i_batch, grad_full_hat, f_i_grad_list, probs, x, y, n):
    
    """
    Function to evaluate SGLD-CV-NUS gradient 
    
    Inputs:    
    key - PRNG 
    theta - parameters (array)
    theta_hat - posterior mode (array)
    gradf_0 - gradient of f_0 (function)
    gradf_i_batch - batched f_i gradients (function)
    grad_full_hat - sum of all f_i gradients at mode (array)
    f_i_grad_list - f_i gradients for all points (array)
    probs - probabilities for each datapoint (array)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    n - subsample size (integer)
    
    Outputs:
    samples, grads - stochastic gradient estimate
    """
       
    N = x.shape[0]
    dim = theta_hat.shape[0]
    key, subkey = random.split(key)
    idx_batch = random.choice(subkey, N, shape=(n,), p=probs)
    
    #calculate stochastic gradient
    probs_sub = jnp.tile(probs[np.array(idx_batch)], reps=(dim, 1)).transpose()

    if y.all() == None:
        term1 =  gradf_i_batch(theta, x[idx_batch,:])/probs_sub
    else:
        term1 = gradf_i_batch(theta, x[idx_batch,:], y[np.array(idx_batch)])/probs_sub
    term2 = f_i_grad_list[idx_batch]/probs_sub
    
    param_grad = gradf_0(theta) + grad_full_hat + (1/n)*jnp.sum(term1-term2, axis=0)
    
    return param_grad


def sgldcv_ps_kernel(key, theta, theta_hat, gradf_0, gradf_i_batch, grad_full_hat, f_i_grad_list, probs, step, x, y, n):
    
    """
    Function to update SGLD-CV-NUS kernel 
    
    Inputs:    
    key - PRNG 
    theta - parameters (array)
    theta_hat - posterior mode (array)
    gradf_0 - gradient of f_0 (function)
    gradf_i_batch - batched f_i gradients (function)
    grad_full_hat - sum of all f_i gradients at mode (array)
    f_i_grad_list - f_i gradients for all points (array)
    probs - probabilities for each datapoint (array)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    n - subsample size (integer)
    
    Outputs:
    samples, grads - stochastic gradient estimate
    """
       
    N = x.shape[0]
    dim = theta_hat.shape[0]
    subkey1, subkey2 = random.split(key)
    idx_batch = random.choice(subkey1, N, shape=(n,), p=probs)
    
    #calculate stochastic gradient
    probs_sub = jnp.tile(probs[np.array(idx_batch)], reps=(dim, 1)).transpose()

    if y.all() == None:
        term1 =  gradf_i_batch(theta, x[idx_batch,:])/probs_sub
    else:
        term1 = gradf_i_batch(theta, x[idx_batch,:], y[np.array(idx_batch)])/probs_sub
    term2 = f_i_grad_list[idx_batch]/probs_sub
    
    param_grad = gradf_0(theta) + grad_full_hat + (1/n)*jnp.sum(term1-term2, axis=0)
    
    #update theta
    theta = theta - (step/2)*param_grad + jnp.sqrt(step)*random.multivariate_normal(key = subkey2, 
                                                                                    mean = jnp.zeros(dim), cov = jnp.eye(dim))
    return theta, param_grad


def sgldcv_ps_sampler(key, gradf_0, gradf_i_batch, post_var, Niter, step, theta_0, theta_hat, x, y, n):
    
    """
    Function to run SGLD-CV-PS sampler with approx probs scheme 
    
    Inputs:
    key - PRNG 
    theta_0 - initial parameter values (array)
    theta_hat - posterior mode (array)
    gradf_0 - gradient of negative log prior (function)
    gradf_i_batch - batched f_i gradients (function)
    post_var - evaluate covariance matrix of Normal approximation (function)
    Niter - number of samples (integer)
    step - step-size tuned for sampler (float)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    n - subsample size (integer)
    
    Outputs:
    samples - parameter values (array)
    grads - stochastic gradients (array)
    run_time - runtimes (array)
    probs - calculated approximate probs (array)
    """
    
    N = x.shape[0]
    dim = theta_hat.shape[0]
    samples = np.zeros((Niter+1, dim))
    grads = np.zeros((Niter, dim))
    run_time = np.zeros(Niter)
    
    x = jnp.array(x)
    if y.all() != None:
        y = jnp.array(y)
        
    cov_mat, gradf_0_hat, grad_full_hat, f_i_grad_list, f_i_hess_list = cvps_preliminaries(theta_hat, gradf_0, 
                                                                                           gradf_i_batch, post_var, x, y)
    
    samples[0,:]=theta_0
    theta = theta_0
   
    probs = approx_probs(theta_hat, f_i_hess_list, cov_mat)
    start_time = time.time()
    for i in range(Niter):
        key, subkey = random.split(key)
        theta, param_grad = sgldcv_ps_kernel(subkey, theta, theta_hat, gradf_0, gradf_i_batch, 
                                             grad_full_hat, f_i_grad_list, probs, step, x, y, n)
        samples[i+1] = theta
        grads[i] = param_grad
        iter_time = time.time()
        run_time[i] = iter_time - start_time
        
    return samples, grads, run_time, probs


def asgld_cv_ps_sampler(key, gradf_0, gradf_i_batch, post_var, Niter, step, theta_0, theta_hat, x, y, cons):
    
    """
    Function to run adaptive SGLD-CV-PS sampler with approximate subsampling 
    
    Inputs:
    key - PRNG 
    theta_0 - initial parameter values (array)
    theta_hat - posterior mode (array)
    gradf_0 - gradient of negative log prior (function)
    gradf_i_batch - batched f_i gradients (function)
    post_var - evaluate covariance matrix of Normal approximation (function)
    Niter - number of samples (integer)
    step - step-size tuned for sampler (float)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    cons - multiplicative constant to scale ||\theta - \theta_hat ||^2 to select n (scalar)
    
    Outputs:
    samples - parameter values (array)
    grads - stochastic gradients (array)
    run_time - runtime of sampler (array)
    n_vec - batch sizes chosen each iteration (array)
    probs - approx probs array (array)
    """
    
    N = x.shape[0]
    dim = theta_hat.shape[0]
    samples = np.zeros((Niter+1, dim))
    grads = np.zeros((Niter, dim))
    run_time = np.zeros(Niter)
    n_vec = np.zeros(Niter)
    
    x = jnp.array(x)
    if y.all() != None:
        y = jnp.array(y)
        
    cov_mat, gradf_0_hat, grad_full_hat, f_i_grad_list, f_i_hess_list = cvps_preliminaries(theta_hat, gradf_0, 
                                                                                           gradf_i_batch, post_var, x, y)
    
    samples[0,:]=theta_0
    theta = theta_0
   
    probs = approx_probs(theta_hat, f_i_hess_list, cov_mat)
    
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
        theta, param_grad = sgldcv_ps_kernel(subkey, theta, theta_hat, gradf_0, 
                                             gradf_i_batch, grad_full_hat, f_i_grad_list, probs, step, x, y, n)
        samples[i+1] = theta
        grads[i] = param_grad
        iter_time = time.time()
        run_time[i] = iter_time - start_time

    return samples, grads, run_time, n_vec, probs
        
