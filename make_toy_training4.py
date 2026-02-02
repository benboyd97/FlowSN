#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--name", type=str,default='base_name')
parser.add_argument("--nn_width", type=int,default=32)
parser.add_argument("--nn_depth", type=int,default=2)
parser.add_argument("--no_flows", type=int,default=4)
parser.add_argument("--max_pat", type=int,default=5)
parser.add_argument("--val_prop", type=float,default=0.1)
parser.add_argument("--batch_size", type=int,default=1024)
parser.add_argument("--epochs", type=int,default=10000)
parser.add_argument('--save_all', action='store_true', help='save old model versions')
parser.add_argument('--make_slurm', action='store_true', help='run_slurm')

args = parser.parse_args()
save_all = args.save_all
name = args.name

import numpy as onp

if name =='base_name':
    name = 'toy_base_w'+str(args.nn_width)+'_d'+str(args.nn_depth)+'_f'+str(args.no_flows)+'_vp'+str(onp.around(args.val_prop,2))+'_bs'+str(args.batch_size)
    if save_all==False:
        name+='_p'+str(args.max_pat)


std_norm = True
restart=False
batch_samps = 20000000
import jax
jax.config.update('jax_enable_x64',True)
import jax.random as random
import copy
import pickle
import subprocess
import numpyro
import numpyro.distributions as dist

from jax import random

import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.scipy.stats import norm

from jax.lib import xla_bridge


from tqdm.notebook import trange
import itertools
import numpy.random as npr
itercount = itertools.count()


itercount = itertools.count()

import jax.numpy as jnp
import jax.random as jr

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow,masked_autoregressive_flow

"""Training loops."""

from collections.abc import Callable

import equinox as eqx
import optax
import paramax
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree, Scalar
from tqdm import tqdm

from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.train_utils import (
    count_fruitless,
    get_batches,
    step,
    train_val_split,
)




@jit
def minmax_fit_and_scale(X):
  max= np.max(X,axis=0)
  min = np.min(X,axis=0)
  X_std = (X - min) / (max-min)
  return X_std,min,max


@jit
def minmax_scale(X,min,max):
  return (X - min) / (max - min)

@jit
def minmax_unscale(X,min,max):
  return X * (max - min) + min

@jit
def std_fit_and_scale(X):
  mu_ = np.mean(X,axis=0)
  std_= np.std(X,axis=0)
  X_std = (X - mu_)/ std_
  return X_std,mu_,std_


@jit
def std_scale(X,mu,std):
  return (X - mu)/std

@jit
def std_unscale(X,mu,std):
  return X*std + mu

_, opt_key= jr.split(jr.key(999))
opt_key,___= jr.split(opt_key)

def sample_model(n_sne,d_hat_s=None,log_x_err=None,log_c_err=None,log_mag_err=None,cov_m_c=None,cov_m_x=None,cov_c_x=None,sel_s=None,m_cut=24,sigma_cut=0.25,a=-0.1,b=-1):

    
    with numpyro.plate("plate_i",n_sne):

        m0  = 25.-numpyro.sample("m0",dist.HalfNormal(0.5))
        
        alpha = numpyro.sample("alpha",dist.Uniform(-0.2,-0.1))
        beta = numpyro.sample("beta",dist.Uniform(2.5,3.5))
        alpha_c = numpyro.sample("alpha_c",dist.Uniform(-0.02,0.0))

        c0 = numpyro.sample("c0",dist.Uniform(-0.1,0.))
        x0 = numpyro.sample("x0",dist.Uniform(-0.6,-0.2))

        sigma_int = numpyro.sample("sigma_int",dist.Uniform(0,0.2))
        sigma_c = numpyro.sample("sigma_c",dist.Uniform(0,0.1))
        sigma_x = numpyro.sample("sigma_x",dist.Uniform(0,1.5))
        
        
        x_s_=numpyro.sample("x_s", dist.Normal(x0, sigma_x))
        
        
        c_s_ = numpyro.sample("c_s", dist.Normal(c0+alpha_c*x_s_,sigma_c))
        
        
        m_s_=numpyro.sample("m_s_", dist.Normal(m0+alpha*x_s_+beta*c_s_, sigma_int))
        
        
        log_x_err = numpyro.sample("log_x_err_s",dist.Normal(-1.5,0.5),obs=log_x_err)
        
        log_c_err = numpyro.sample("log_c_err_s",dist.Normal(-3.5,0.3),obs=log_c_err)
        
        log_mag_err = numpyro.sample("log_mag_err_s",dist.Normal(0.1*(m_s_-56),0.6),obs=log_mag_err)
        

        cov_m_c=numpyro.sample("cov_m_c",dist.Normal(0.000520+0.288792999*jnp.exp(log_mag_err)**2,jnp.exp(log_mag_err)**2*0.125979),obs=cov_m_c)

        cov_m_x=numpyro.sample("cov_m_x",dist.Normal(0.0008097+0.03835550*jnp.exp(log_x_err)**2,jnp.exp(log_x_err)**2*0.02032),obs=cov_m_x)

        cov_c_x=numpyro.sample("cov_c_x",dist.Normal(0.000168655+0.01358504*jnp.exp(log_x_err)**2,jnp.exp(log_x_err)**2*0.011840861),obs=cov_c_x)

        W_s = jnp.array([[jnp.exp(log_mag_err)**2,cov_m_c,cov_m_x],
                         [cov_m_c,jnp.exp(log_c_err)**2,cov_c_x],
                         [cov_m_x,cov_c_x,jnp.exp(log_x_err)**2]]).T
        
        

        
        d_hat_s=numpyro.sample("d_hat_s", dist.MultivariateNormal(jnp.column_stack((m_s_,c_s_,x_s_)),W_s),obs=d_hat_s)
        

        p_s=norm.cdf(-(d_hat_s[:,0]+a*d_hat_s[:,2]+b*d_hat_s[:,1]),loc=-m_cut,scale=sigma_cut)
        
        sel_s=numpyro.sample("sel_s",dist.Bernoulli(p_s),obs=sel_s)

def sample_model2(n_sne,d_hat_s=None,log_x_err=None,log_c_err=None,log_mag_err=None,cov_m_c=None,cov_m_x=None,cov_c_x=None,sel_s=None,m_cut=24,sigma_cut=0.25,a=-0.1,b=-1):


    
    with numpyro.plate("plate_i",n_sne):

        m0  = numpyro.sample("m0",dist.Uniform(17,m_cut))
        
        alpha = numpyro.sample("alpha",dist.Uniform(-0.2,-0.1))
        beta = numpyro.sample("beta",dist.Uniform(2.5,3.5))
        alpha_c = numpyro.sample("alpha_c",dist.Uniform(-0.02,0.0))

        c0 = numpyro.sample("c0",dist.Uniform(-0.1,0.))
        x0 = numpyro.sample("x0",dist.Uniform(-0.6,-0.2))

        sigma_int = numpyro.sample("sigma_int",dist.Uniform(0,0.2))
        sigma_c = numpyro.sample("sigma_c",dist.Uniform(0,0.1))
        sigma_x = numpyro.sample("sigma_x",dist.Uniform(0,1.5))

        
        
        x_s_=numpyro.sample("x_s", dist.Normal(x0, sigma_x))
        
        
        c_s_ = numpyro.sample("c_s", dist.Normal(c0+alpha_c*x_s_,sigma_c))
        
        
        m_s_=numpyro.sample("m_s_", dist.Normal(m0+alpha*x_s_+beta*c_s_, sigma_int))
        
        
        log_x_err = numpyro.sample("log_x_err_s",dist.Normal(-1.5,0.5),obs=log_x_err)
        
        log_c_err = numpyro.sample("log_c_err_s",dist.Normal(-3.5,0.3),obs=log_c_err)
        
        log_mag_err = numpyro.sample("log_mag_err_s",dist.Normal(0.1*(m_s_-56),0.6),obs=log_mag_err)
        

        cov_m_c=numpyro.sample("cov_m_c",dist.Normal(0.000520+0.288792999*jnp.exp(log_mag_err)**2,jnp.exp(log_mag_err)**2*0.125979),obs=cov_m_c)

        cov_m_x=numpyro.sample("cov_m_x",dist.Normal(0.0008097+0.03835550*jnp.exp(log_x_err)**2,jnp.exp(log_x_err)**2*0.02032),obs=cov_m_x)

        cov_c_x=numpyro.sample("cov_c_x",dist.Normal(0.000168655+0.01358504*jnp.exp(log_x_err)**2,jnp.exp(log_x_err)**2*0.011840861),obs=cov_c_x)

        W_s = jnp.array([[jnp.exp(log_mag_err)**2,cov_m_c,cov_m_x],
                         [cov_m_c,jnp.exp(log_c_err)**2,cov_c_x],
                         [cov_m_x,cov_c_x,jnp.exp(log_x_err)**2]]).T
        
        

        
        d_hat_s=numpyro.sample("d_hat_s", dist.MultivariateNormal(jnp.column_stack((m_s_,c_s_,x_s_)),W_s),obs=d_hat_s)
        

        p_s=norm.cdf(-(d_hat_s[:,0]+a*d_hat_s[:,2]+b*d_hat_s[:,1]),loc=-m_cut,scale=sigma_cut)
        
        sel_s=numpyro.sample("sel_s",dist.Bernoulli(p_s),obs=sel_s)

        
key, rng= jr.split(jr.key(0))

from numpyro.infer import MCMC, NUTS, Predictive


for reps__ in range(20):

    prior_predictive = Predictive(sample_model, num_samples=1)

    key, rng= jr.split(key)

    prior_predictions = prior_predictive(key,18000000)
    sel_sim =prior_predictions["sel_s"][0,:]

    log_x_err = prior_predictions["log_x_err_s"][0,:][sel_sim==1]
    log_c_err = prior_predictions["log_c_err_s"][0,:][sel_sim==1]
    log_mag_err = prior_predictions["log_mag_err_s"][0,:][sel_sim==1]

    cov_m_c = prior_predictions["cov_m_c"][0,:][sel_sim==1]
    cov_m_x = prior_predictions["cov_m_x"][0,:][sel_sim==1]
    cov_c_x= prior_predictions["cov_c_x"][0,:][sel_sim==1]

    d_hat_sim = prior_predictions["d_hat_s"][0,:,:][sel_sim==1]

    m0 = 25.-prior_predictions["m0"][0,:][sel_sim==1]
    c0 = prior_predictions["c0"][0,:][sel_sim==1]
    x0 = prior_predictions["x0"][0,:][sel_sim==1]

    alpha = prior_predictions["alpha"][0,:][sel_sim==1]
    beta = prior_predictions["beta"][0,:][sel_sim==1]
    alpha_c = prior_predictions["alpha_c"][0,:][sel_sim==1]

    sigma_int = prior_predictions["sigma_int"][0,:][sel_sim==1]
    sigma_c = prior_predictions["sigma_c"][0,:][sel_sim==1]
    sigma_x = prior_predictions["sigma_x"][0,:][sel_sim==1]

    X1= jnp.append(d_hat_sim, jnp.column_stack((m0,c0,x0,jnp.log(sigma_int),jnp.log(sigma_c),jnp.log(sigma_x),alpha,beta,alpha_c,log_mag_err, log_c_err,log_x_err,cov_m_c,cov_m_x,cov_c_x)),axis=1)[:int(0.9*0.05*batch_samps),:]

    key, rng= jr.split(key)

    prior_predictive = Predictive(sample_model2, num_samples=1)

    prior_predictions = prior_predictive(key,1500000)
    sel_sim =prior_predictions["sel_s"][0,:]

    log_x_err = prior_predictions["log_x_err_s"][0,:][sel_sim==1]
    log_c_err = prior_predictions["log_c_err_s"][0,:][sel_sim==1]
    log_mag_err = prior_predictions["log_mag_err_s"][0,:][sel_sim==1]

    cov_m_c = prior_predictions["cov_m_c"][0,:][sel_sim==1]
    cov_m_x = prior_predictions["cov_m_x"][0,:][sel_sim==1]
    cov_c_x= prior_predictions["cov_c_x"][0,:][sel_sim==1]

    d_hat_sim = prior_predictions["d_hat_s"][0,:,:][sel_sim==1]

    m0 = prior_predictions["m0"][0,:][sel_sim==1]
    c0 = prior_predictions["c0"][0,:][sel_sim==1]
    x0 = prior_predictions["x0"][0,:][sel_sim==1]

    alpha = prior_predictions["alpha"][0,:][sel_sim==1]
    beta = prior_predictions["beta"][0,:][sel_sim==1]
    alpha_c = prior_predictions["alpha_c"][0,:][sel_sim==1]

    sigma_int = prior_predictions["sigma_int"][0,:][sel_sim==1]
    sigma_c = prior_predictions["sigma_c"][0,:][sel_sim==1]
    sigma_x = prior_predictions["sigma_x"][0,:][sel_sim==1]

    X2= jnp.append(d_hat_sim, jnp.column_stack((m0,c0,x0,jnp.log(sigma_int),jnp.log(sigma_c),jnp.log(sigma_x),alpha,beta,alpha_c,log_mag_err, log_c_err,log_x_err,cov_m_c,cov_m_x,cov_c_x)),axis=1)[:int(0.1*0.05*batch_samps),:]

    X_ = jnp.append(X1,X2,axis=0)
    if reps__==0:
       X = X_
    else:
       X = jnp.append(X,X_,axis=0)

    print(X.shape)


print(X.shape)
np.save('toy_training4.npy',X) 
key, rng= jr.split(key)

from matplotlib import pyplot as plt

print(np.max(X[:,3]))
print(len(X[X[:,3]>25.1,3])/len(X[:,3]))
print(len(X[X[:,3]>25.1,3])/len(X[:,3]))
print(len(X[X[:,3]<20.1,3])/len(X[:,3]))
plt.hist(X[:,3],bins=100)
plt.show()

