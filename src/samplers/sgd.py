#!/usr/bin/env python3
"""

SGD / ADAM sampler 

"""
import jax.numpy as jnp
from jax import jit, random
import numpy as np
import time
from jax.experimental import optimizers

def adam(key, gradf_0, gradf_i_batch, Niter_sgd, theta_0, x, y, n, step_sgd, replacement):
    
    """
    Function to conduct adam optimisation to find posterior mode
    
    Inputs:    
    key - PRNG 
    gradf_0 - gradient of negative log prior (function)
    gradf_i_batch - batched f_i gradients (function)
    Niter_sgd - number of samples (integer)
    theta_0 - initial parameters (array)
    x - observations (array)
    y - response variables / array of None if not in model  (array)
    n - subsample size (integer)
    step_sgd - step-size for SGD (float)
    replacement - boolean (True or False) for sampling with/with-out replacement
    
    Outputs:
    theta_hat - posterior mode (array)
    samples_sgd - trace of optimiser (array)
    run_time - runtime of optimiser (array)
    
    """
    N = np.shape(x)[0]
    dim = theta_0.shape[0] #dim is number of parameters
    x = jnp.array(x)
    y = jnp.array(y)
    
    samples_sgd = np.zeros((Niter_sgd+1, dim)) #store sgd samples 
    run_time = np.zeros(Niter_sgd) #store runtimes 
    samples_sgd[0] = np.array(theta_0)
    
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_sgd)  #choose optimiser settings
    opt_state = opt_init(theta_0) #set initial state
            
    @jit
    def update(i, opt_state, x_batch, y_batch, N):
        params = get_params(opt_state) #extract current parameter
        grad_list = gradf_i_batch(params, x_batch, y_batch)     
        grad = N*jnp.mean(grad_list, axis = 0) + gradf_0(params)  #stochastic (subsampled) gradient
        return opt_update(i, grad, opt_state) 

    #run optimiser
    start_time = time.time()
    for i in range(Niter_sgd):
        key, subkey = random.split(key)
        idx_batch = random.choice(subkey, N, shape=(n,), replace=replacement)
        x_batch = x[idx_batch,:]
        y_batch =y[idx_batch]
        opt_state=update(i, opt_state, x_batch, y_batch, N)
        samples_sgd[i+1]= get_params(opt_state)

        #store runtimes
        iter_time = time.time()-start_time
        run_time[i]=iter_time
    
    #obtain posterior mode
    theta_hat = samples_sgd[Niter_sgd]
        
    return theta_hat, samples_sgd, run_time



def adam_x(key, gradf_0, gradf_i_batch, Niter_sgd, theta_0, x, n, step_sgd, replacement):
    
    """
    Function to conduct adam optimisation to find posterior mode with response y 
    
    Inputs:    
    key - PRNG 
    gradf_0 - gradient of negative log prior (function)
    gradf_i_batch - batched f_i gradients (function)
    Niter_sgd - number of samples (integer)
    theta_0 - initial parameters (array)
    x - observations (array)
    n - subsample size (integer)
    step_sgd - step-size for SGD
    replacement - boolean (True or False) for sampling with/with-out replacement
    
    Outputs:
    theta_hat - posterior mode (array)
    samples_sgd - trace of optimiser (array)
    run_time - runtime of optimiser (array)
    
    """
    N = np.shape(x)[0]
    dim = theta_0.shape[0] #dim is number of parameters
    x = jnp.array(x)
    
    samples_sgd = np.zeros((Niter_sgd+1, dim)) #store sgd samples 
    run_time = np.zeros(Niter_sgd) #store runtimes 
    samples_sgd[0] = np.array(theta_0)
    
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)  #choose optimiser settings
    opt_state = opt_init(theta_0) #set initial state
            
    @jit
    def update(i, opt_state, x_batch, N):
        params = get_params(opt_state) #extract current parameter
        grad_list = gradf_i_batch(params, x_batch)     
        grad = N*jnp.mean(grad_list, axis = 0) + gradf_0(params)  #stochastic (subsampled) gradient
        return opt_update(i, grad, opt_state) 

    #run optimiser
    start_time = time.time()
    for i in range(Niter_sgd):
        key, subkey = random.split(key)
        idx_batch = random.choice(subkey, N, shape=(n,), replace=replacement)
        x_batch = x[idx_batch,:]
        opt_state=update(i, opt_state, x_batch, N)
        samples_sgd[i+1]= get_params(opt_state)

    #store runtimes
        iter_time = time.time()-start_time
        run_time[i]=iter_time
    
    #obtain posterior mode
    theta_hat = samples_sgd[Niter_sgd]
        
    return theta_hat, samples_sgd, run_time
