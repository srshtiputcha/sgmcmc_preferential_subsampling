#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Metrics for experiments

"""
import numpy as np
from jax import jit, vmap
import jax.numpy as jnp
from jax import lax
import models.logistic_regression.logistic_regression as lr

##############################################################################
### Kernel Stein Discrepancy (KSD) metric

@jit
def k_0_fun(parm1, parm2, gradlogp1, gradlogp2, c=1., beta=-0.5):
    """
    KSD kernel with the 2 norm
    """
    diff = parm1-parm2
    dim = parm1.shape[0]
    base = (c**2 + jnp.dot(diff, diff))
    term1 = jnp.dot(gradlogp1,gradlogp2)*base**beta
    term2 = -2*beta * jnp.dot(gradlogp1, diff) * base**(beta-1)
    term3 = 2*beta * jnp.dot(gradlogp2, diff) * base**(beta-1)
    term4 = -2*dim*beta*(base**(beta-1))
    term5 = -4*beta* (beta-1)*base**(beta-2)*jnp.sum(jnp.square(diff))
    return term1 + term2 + term3 + term4 + term5

batch_k_0_fun_rows = jit(vmap(k_0_fun, in_axes=(None,0,None,0,None,None)))

@jit
def imq_KSD(samples, grads):
    """
    KSD with imq kernel
    
    Inputs: 
    samples - MCMC samples obtained for all parameters (array)
    grads - corresponding stochastic gradient (array)
    
    Outputs:
    result - KSD value
    """
    c, beta = 1., -0.5
    N = samples.shape[0]

    def body_ksd(le_sum, x):
        my_sample, my_grad = x
        le_sum += jnp.sum(batch_k_0_fun_rows(my_sample, samples, my_grad, grads, c, beta))
        return le_sum, None

    le_sum, _ = lax.scan(body_ksd, 0., (samples, grads))
    result = jnp.sqrt(le_sum)/N
    return result

#############################################################################
######### Logistic regression metrics

def logloss(theta_current, x_test, y_test):
    
    """
    Function to evaluate the log-loss of the LR model
    
    Args:
    theta_current - current value of theta (array)
    x_test - covariates in test dataset (array)
    y_test - responses variables in test data (array)
    
    Outputs:
    ll - logloss value (float)
    
    """
    
    theta = theta_current
    
    Ntest = x_test.shape[0]
    ll = lr.log_p(theta, x_test, y_test) / Ntest
    
    return ll

    
###########################################################################
#### Bivariate Gaussian metrics
    
def kullbeck_liebler(mp, mq, Sp,Sq):
    
    """
    Calculates KL-divergence from Gaussian target posterior p to approximating distribution q
    
    Inputs:
    mp - mean vector of p (mean)
    mq - mean vector of q (mean)
    Sp - covariance of p (matrix)
    Sq - covariance of q (matrix)
    
    Outputs:
    result - KL divergence value (float)
    """
    
    k = mp.shape[0]
    inv_Sq = np.linalg.inv(Sq)
    diff = mp - mq
    
    term1 = np.trace(np.matmul(inv_Sq, Sp))
    term2 = diff.T @ np.linalg.inv(Sq) @ diff
    term3 = np.log(np.linalg.det(Sq)/np.linalg.det(Sp))
    
    result = 0.5 * (term1 + term2 + term3 - k)
    
    return result
    
