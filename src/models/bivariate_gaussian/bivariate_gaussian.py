#!/usr/bin/env python3
"""

Bivariate Gaussian model 

"""
#import modules
import jax.numpy as jnp
from jax import jit, grad, vmap, hessian, scipy

from jax.config import config
config.update("jax_enable_x64", True) #enforce float64 precision

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
    result = (-1) * scipy.stats.multivariate_normal.logpdf(x = theta, mean = mu_0, cov= lambda_0) 
    return result

gradf_0 = jit(grad(f_0, argnums=0))
hessf_0 = jit(hessian(f_0, argnums=0))

@jit
def f_i(theta, x_val):
    
    """
    Function to evaluate the negative log-likelihood of i-th observation 
    
    Args:   
    theta - parameter (array)
    x_val - i-th datapoint (array)
    
    Outputs: 
    result - (-1) * log likelihood of datapoint evaluated at current theta (scalar)
    """
    sigma_x=jnp.array([[1,-0.5],[-0.5, 1.5]]) #model covariance
    result = (-1) * scipy.stats.multivariate_normal.logpdf(x = x_val, mean = theta, cov= sigma_x)
    return result

#calculates f_i's for several datapoints 
f_i_batch = jit(vmap(f_i, in_axes = (None,0), out_axes=0))

#gradient of i-th log density, f_i
gradf_i = jit(grad(f_i, argnums= 0)) #take gradient with respect to parameters

#calculates gradients of f_i's for several datapoints and outputs d dimensional array
gradf_i_batch = jit(vmap(gradf_i, in_axes = (None, 0), out_axes = 0))

#hessian of f_i
hessf_i = jit(hessian(f_i, argnums= 0)) #take gradient with respect to parameters

#hessian of f_i for several datapoints and outputs array
hessf_i_batch = jit(vmap(hessf_i, in_axes = (None, 0), out_axes = 0))

@jit
def gradf(theta, x):
    
    """
    Function to evaluate the negative log posterior on full data
    
    Args:
    theta - parameter (array)
    x - data (array)
    
    Outputs:
    result - (-1) * log posterior at current theta (scalar)

    """
    N = jnp.shape(x)[0]
    term1 = gradf_0(theta)
    term2 = jnp.int64(N)*jnp.mean(gradf_i_batch(theta, x), axis=0)
    return term1+term2

@jit
def post_var(theta, x):
    
    """
    Function to evaluate to inverse of the hessian of the negative log posterior
    
    Args:
    theta - parameter (array)
    x - data (array)
    
    Outputs:
    sigma_hat - inverse of the hessian of the negative log posterior
    
    """
    term1 = hessf_0(theta) #hessian of f_0
    f_i_hess_list = hessf_i_batch(theta, x) #hessians of all f_i 
    term2=jnp.sum(f_i_hess_list, axis=0) 
    obs_inf = term1 + term2 #observed information matrix
    
    cov_mat = jnp.linalg.inv(obs_inf)
    
    return cov_mat, f_i_hess_list
    
    
    
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
