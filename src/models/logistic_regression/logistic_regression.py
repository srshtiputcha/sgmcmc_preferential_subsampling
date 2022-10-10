#!/usr/bin/env python3
"""

Logistic Regression model 

"""
#import modules
import jax.numpy as jnp
from jax import jit, grad, vmap, hessian, scipy
import numpy as np

@jit
def f_0(theta):
    
    """
    Function to evaluate the negative log prior, f_0
    
    Args: 
    theta - parameter (array)
    
    Outputs: 
    result - (-1) * log prior evaluated at theta (scalar)
    
    """
    dim = jnp.size(theta) #size of parameter space 
    mu_0 = jnp.zeros(dim) #prior mean 
    lambda_0 = jnp.diag(jnp.array([10]*dim)) #prior covariance
    result = (-1.0) * scipy.stats.multivariate_normal.logpdf(x = theta, mean = mu_0, cov= lambda_0) 
    
    return result

gradf_0 = grad(f_0, argnums=0)
hessf_0 = jit(hessian(f_0, argnums=0))

@jit
def f_i(theta, x_val, y_val):
    
    """
    Function to evaluate the negative log-likelihood of i-th observation 
    
    Args:   
    theta - parameters (array)
    x_val - i-th datapoint (array)
    y_val - i-th response variables (integer)
    
    Outputs: 
    result - (-1) * log likelihood of datapoint evaluated at current theta (scalar)
    """

    c = scipy.special.expit(jnp.matmul(x_val, theta))
    result = (-1.0) * jnp.nansum(y_val*jnp.log(c) + (1.0-y_val)*jnp.log(1.0-c))
        
    return result


#calculates f_i's for several datapoints 
f_i_batch = jit(vmap(f_i, in_axes = (None, 0,0), out_axes = 0))

def log_p(theta, x, y):
    """
    Function to evaluate the negative log-likelihood for a given sample over all datapoints
    
    Args:
    theta - current sample (array)
    x - array of x observations
    y - array of y observations
    
    Outputs:
    log_p - log-likelihood of all test datapoints on current MCMC sample (scalar)
    """
    
    log_liks = np.array(f_i_batch(theta, x, y))
    log_p = np.sum(log_liks)
    return log_p
    

#gradient of i-th log density, f_i
gradf_i = jit(grad(f_i, argnums= 0)) #take gradient with respect to parameters

#calculates gradients of f_i's for several datapoints and outputs d dimensional array
gradf_i_batch = jit(vmap(gradf_i, in_axes = (None, 0,0), out_axes = 0))

#hessian of f_i
hessf_i = jit(hessian(f_i, argnums= 0)) #take gradient with respect to parameters

#hessian of f_i for several datapoints and outputs array
hessf_i_batch = jit(vmap(hessf_i, in_axes = (None, 0,0), out_axes = 0))

@jit
def gradf(theta, x, y):
    
    """
    Function to evaluate the negative log posterior on full data
    
    Args:
    theta - parameter (array)
    x - data (array)
    y - vector of responses (array)
    
    Outputs:
    result - (-1) * log posterior at current theta (scalar)

    """
    N = jnp.shape(x)[0]
    term1 = gradf_0(theta)
    term2 = jnp.int64(N)*jnp.mean(gradf_i_batch(theta, x, y), axis=0)
    return term1+term2

@jit
def post_var(theta, x, y):
    
    """
    Function to evaluate to inverse of the hessian of the negative log posterior
    
    Args:
    theta - parameter (array)
    x - data (array)
    y - vector of responses (array)
    
    Outputs:
    sigma_hat - inverse of the hessian of the negative log posterior
    
    """
    term1 = hessf_0(theta) #hessian of f_0
    f_i_hess_list = hessf_i_batch(theta, x, y) #hessians of all f_i 
    term2=jnp.sum(f_i_hess_list, axis=0) 
    obs_inf = term1 + term2 #observed information matrix
    
    cov_mat = jnp.linalg.inv(obs_inf)
    
    return cov_mat, f_i_hess_list
