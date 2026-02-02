#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--rep", type=int,default=0)
parser.add_argument("--name", type=str,default='base_name')
parser.add_argument("--nn_width", type=int,default=32)
parser.add_argument("--nn_depth", type=int,default=2)
parser.add_argument("--no_flows", type=int,default=4)
parser.add_argument('--model_type',type=str,default='flow')
parser.add_argument('--cmb', action='store_true', help='CMB prior')



vary_cmb = True
sigma_Rcmb = 0.007
R_cmb_obs_default=1.7579698042257326


args = parser.parse_args()
rep_ = args.rep
name = args.name
model_type=args.model_type
cmb_bool = args.cmb


import os
std_norm = True
wCDM_bool = True
ZMIN = 0.01
ZMAX = 1.2

H0 = 70.

import jax
jax.config.update('jax_enable_x64',True)

import jax.random as random
from jax import jit

import pickle

import numpyro
numpyro.set_host_device_count(4)
from SkewNormalPlus import SkewNormalPlus as snp
from SkewNormalPlus3DCov_FULL import SkewNormalPlus3D
from NaiveCov_FULL import Naive


from jax import random
import jax_cosmo as jc
from jax_cosmo import Cosmology, background
import numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.scipy.stats import norm

from jax.lib import xla_bridge


from tqdm.notebook import trange
import itertools
import numpy.random as npr
itercount = itertools.count()

import logging
itercount = itertools.count()
from matplotlib import pyplot as plt


import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

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





import wcosmo
import numpy.typing

if vary_cmb:
    key = jax.random.PRNGKey(rep_)  
    key_cmb, _= random.split(key)
    cmb_draw = jax.random.normal(key_cmb)
    R_cmb_obs=R_cmb_obs_default+cmb_draw*sigma_Rcmb
else:
    R_cmb_obs=R_cmb_obs_default



def set_backend(backend):
    from importlib import import_module
    np_modules = dict(
        numpy="numpy",
        jax="jax.numpy",
        cupy="cupy",
    )
    linalg_modules = dict(
        numpy="scipy.linalg",
        jax="jax.scipy.linalg",
        cupy="cupyx.scipy.linalg",
    )
    setattr(wcosmo.wcosmo, "xp", import_module(np_modules[backend]))
    setattr(wcosmo.utils, "xp", import_module(np_modules[backend]))
    toeplitz = getattr(import_module(linalg_modules[backend]), "toeplitz")
    setattr(wcosmo.utils, "toeplitz", toeplitz)


set_backend("jax")


@jit
def time_jax_comoving_distance(jdata,omega_m,w):
    return wcosmo.FlatwCDM(H0, omega_m[0], w[0]).comoving_distance(jdata)


@jit
def jax_dvcdz(jdata,omega_m,w):
    return wcosmo.FlatwCDM(H0, omega_m[0], w[0]).differential_comoving_volume(jdata)

from jax.scipy.integrate import trapezoid

import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
z_values = jnp.linspace(ZMIN, ZMAX, 1000)  # Points to evaluate the PDF
mask = np.ones((len(z_values),len(z_values)))
for i in range(len(z_values)):
    mask[i,i+1:]=0
mask = jnp.array(mask)

@jax.jit
def integrate_row(pdf_row, x_row):
    cdf = trapezoid(pdf_row, x=x_row, axis=-1)  # Integrate cumulatively
    return cdf

@jax.jit
def vmap_trap(pdf, x=None):
    integ = jax.vmap(integrate_row, in_axes=(0, None))(pdf, x)
    return integ

@jax.jit
def cdf(pdf_values):
    pdf_values_batch = jnp.repeat(pdf_values.reshape(-1,1),len(pdf_values),axis=1).T

    pdf_values_batch = jnp.array(pdf_values_batch*mask)


    cdf_values = vmap_trap(pdf_values_batch, z_values)
    
    return cdf_values


# Define the PDF for the Beta(2, 1) distribution
@jax.jit
def pdf(z,omega_m,w,beta_rate):
    prob=(1+z)**(beta_rate)/(1+z)*jax_dvcdz(z,jnp.array([omega_m]),jnp.array([w]))
    all_prob=(1+z_values)**(beta_rate)/(1+z_values)*jax_dvcdz(z_values,jnp.array([omega_m]),jnp.array([w]))
    integ=integrate_row(all_prob, z_values)
    return prob/integ


@jax.jit
def find_ids(b,u):
    u=u.reshape(u.shape[0],u.shape[1],1)
    
    u = jnp.repeat(u,b.shape[1],axis=2)
    
    b = b.reshape(1,b.shape[0],b.shape[1])
    
    b = jnp.repeat(b,u.shape[0],axis=0)
    
    return jnp.absolute(u-b).argmin(axis=-1)



@jax.jit
def interp(x, xp, fp):

        
    ids=find_ids(xp,x)

    
    expanded_ids = jnp.expand_dims(ids, axis=1) 
        
    m,k=ids.shape


    xi = xp[jnp.arange(k), expanded_ids].reshape(expanded_ids.shape[0],expanded_ids.shape[2])

    s = jnp.sign(x-xi).astype(int).reshape(expanded_ids.shape[0],1,expanded_ids.shape[2])

    fi = fp[jnp.arange(k), expanded_ids].reshape(expanded_ids.shape[0],expanded_ids.shape[2])
    
    a = (fp[jnp.arange(k), expanded_ids+  s].reshape(expanded_ids.shape[0],expanded_ids.shape[2]) - fi) / (
    xp[jnp.arange(k), expanded_ids+ s].reshape(expanded_ids.shape[0],expanded_ids.shape[2]) - xi)
    b = fi - a * xi
    return a * x + b


def sample_redshifts(size,key,Om0=0.315,w0=-1,beta_rate=1.5):
    rng, _= random.split(key)
    u_samps=random.uniform(rng,(size,)).reshape(1,-1)
    samps=interp(u_samps,cdf(pdf(z_values,Om0,w0,beta_rate)).reshape(1,-1),z_values.reshape(1,-1))[0,:]
    return samps



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

if model_type=='flow':

    if std_norm:
        file_=np.load(name+'_std.npz')
        mu_ = file_['mu']
        std_ = file_['std']
        add_ = jnp.sum(jnp.log(std_[:3]))
    else:
        file_=np.load(name+'_minmax.npz')
        min_ = file_['min']
        max_ = file_['max']
        add_ = jnp.sum(jnp.log(max_-min_)[:3])





    from flowjax.experimental.numpyro import distribution_to_numpyro
    key, subkey= jr.split(jr.key(2))


    skel_flow =masked_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(3)),
        cond_dim=15,
    nn_activation= jax.nn.gelu,flow_layers=args.no_flows,
    nn_width=args.nn_width,
    nn_depth=args.nn_depth,
    )



    flow_model = eqx.tree_deserialise_leaves(name+'.eqx',skel_flow)

    from numpyro.distributions import (
        Distribution,
        constraints)
    from numpyro.distributions.util import is_prng_key, promote_shapes, validate_sample

    class FlowSNP3D(Distribution):
        arg_constraints = {"m0": constraints.real,"c0": constraints.real,"x0": constraints.real,
        "sigma_int": constraints.real,"sigma_c": constraints.real,"sigma_x": constraints.real,
        "alpha": constraints.real,"beta": constraints.real,"alpha_c": constraints.real, "W_mm": constraints.real,"W_cc": constraints.real,"W_xx": constraints.real,
        "W_mc": constraints.real,"W_mx": constraints.real,"W_cx": constraints.real,
        "m_cut": constraints.real, "sigma_cut": constraints.real,"a":constraints.real,"b":constraints.real}

        support = constraints.real
        reparametrized_params = ["m0", "c0","x0","sigma_int","sigma_c","sigma_x",
        "alpha","beta","alpha_c","W_mm","W_cc","W_xx","W_mc","W_mx","W_cx","m_cut", "sigma_cut","a","b"]

        def __init__(self,m0,c0,x0,sigma_int,sigma_c,sigma_x,alpha,beta,alpha_c,W_mm,W_cc,W_xx,W_mc,W_mx,W_cx,m_cut,sigma_cut,a,b,*,validate_args=False,res=1000):

            self.m0,self.c0,self.x0,self.sigma_int,self.sigma_c,self.sigma_x,self.alpha,self.beta,self.alpha_c,self.W_mm,self.W_cc,self.W_xx,self.W_mc,self.W_mx,self.W_cx,self.m_cut,self.sigma_cut,self.a,self.b=(m0,c0,x0,sigma_int,sigma_c,sigma_x,alpha,beta,alpha_c,W_mm,W_cc,W_xx,W_mc,W_mx,W_cx,m_cut,sigma_cut,a,b)
            
    
            super(FlowSNP3D, self).__init__(
                batch_shape=(m0.shape[0],),
                event_shape=(3,),
                validate_args=validate_args,
            )
        


        def sample(self,key, sample_shape=()):
            assert is_prng_key(key)

                    
            X=jnp.column_stack((self.m0,self.c0,self.x0,jnp.log(self.sigma_int),
                                                jnp.log(self.sigma_c),jnp.log(self.sigma_x),self.alpha,
                                                self.beta,self.alpha_c,np.log(self.W_mm**0.5),jnp.log(self.W_cc**0.5),jnp.log(self.W_xx**0.5),self.W_mc,self.W_mx,self.W_cx))


            X= std_scale(X,mu_[3:],std_[3:])

            samp = std_unscale(flow_model.sample(key,(1,),condition=X),mu_[:3],std_[:3])

            X= std_unscale(X,mu_[3:],std_[3:])

            samp = samp.at[:,:3].set(samp[:,:3]+X[:,:3])[:,:3]


            return samp.reshape(sample_shape[0],sample_shape[1],3,order='F')

        @jax.jit
        def log_prob(self, value):
            
            no_samps = 1
            no_obj = value.size
            
            X=jnp.append(value,jnp.column_stack((self.m0,self.c0,self.x0,jnp.log(self.sigma_int),
                                                jnp.log(self.sigma_c),jnp.log(self.sigma_x),self.alpha,
                                                self.beta,self.alpha_c,jnp.log(self.W_mm**0.5),jnp.log(self.W_cc**0.5),jnp.log(self.W_xx**0.5),self.W_mc,self.W_mx,self.W_cx)),axis=1)

            X = X.at[:,:3].set(X[:,:3]-X[:,3:6])

            X = std_scale(X,mu_,std_)
        
            return flow_model.log_prob(X[:,:3],condition=X[:,3:]) - add_
            




    # In[ ]:


import numpy as np
from matplotlib import pyplot as plt
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.contrib.tfp.mcmc import RandomWalkMetropolis
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from jax import random
import jax

from jax.scipy.stats import norm
from jax.scipy.special import ndtri
#assert numpyro.__version__.startswith("0.11.0")

numpyro.set_host_device_count(4)
jax.config.update('jax_enable_x64',True)


import jax_cosmo as jc
from jax_cosmo import Cosmology, background


from numpyro import sample

import numpyro.distributions as dist

import jax.numpy as jnp

from numpyro.distributions.truncated import TruncatedDistribution 


    

def sample_model(z_s,d_hat_s=None,log_x_err=None,log_c_err=None,log_mag_err=None,cov_m_c=None,cov_c_x=None,cov_m_x=None,sel_s=None,M0=-19.365,sigma_int=0.1
            ,alpha =-0.14,alpha_c=-0.008,beta = 3.1, x0 = -0.432, sigma_x = 1.124, c0= -0.061, sigma_c = 0.065,
            h=H0/100,Om0=0.315,w=-1,m_cut=24,sigma_cut=0.25,a=-0.1,b=-1,dust=False,sigma_pec=300):

        cosmo_jax = Cosmology(Omega_c=Om0, h=h, w0=w, Omega_b=0, n_s= 0.96, sigma8=200000, Omega_k=0, wa=0)


        n_sne=len(z_s)
        
        
        with numpyro.plate("plate_i",n_sne):
            
            d_s = time_jax_comoving_distance(z_s,jnp.array([Om0]),jnp.array([w]))
            mu_s=5*jnp.log10(d_s*(1+z_s)*1e6/10)
        
            x_s_=numpyro.sample("x_s", dist.Normal(x0, sigma_x))
            
            
            c_s_ = numpyro.sample("c_s", dist.Normal(c0+alpha_c*x_s_,sigma_c))
            
            
            m_s_=numpyro.sample("m_s_", dist.Normal(M0+mu_s+alpha*x_s_+beta*c_s_, sigma_int))
            
            
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


        

        

    
    
def mcmc_model(z_s,data_s=None,data_err_s=None,h=H0/100,m_cut=24,sigma_cut=0.25,a=-0.1,b=-1,dust=False,sigma_pec=300,wCDM=True,model='analytical'):

 
    if wCDM:
        Om0=numpyro.sample('Om0',dist.Uniform(0,1))
        w = numpyro.sample('w',dist.Uniform(-2,2))
        Omde= 1-Om0
    else:
        Om0=numpyro.sample('Om0',dist.Uniform(-2,2))
        Omde = numpyro.sample('Omde',dist.Uniform(-2,2))

        w=-1
        
    
    alpha = numpyro.sample('alpha', dist.Uniform(-0.2,-0.1))
    beta = numpyro.sample('beta', dist.Uniform(2.5,3.5))
    alpha_c = numpyro.sample('alpha_c', dist.Uniform(-0.02,0.0))
    
    M0=numpyro.sample('M0', dist.ImproperUniform(dist.constraints.real,(),event_shape=()))
    c0 = numpyro.sample("c0_int",dist.Uniform(-0.1,0.))
    x0 = numpyro.sample("x0",dist.Uniform(-0.6,-0.2))

    sigma_int = numpyro.sample("sigma_int", dist.HalfNormal(1))
    sigma_c = numpyro.sample("sigma_c", dist.HalfNormal(1))
    sigma_x = numpyro.sample("sigma_x", dist.HalfNormal(2))


    cosmo_jax = Cosmology(Omega_c=Om0, h=h, w0=w, Omega_b=0, n_s= 0.96, sigma8=200000, Omega_k=1-(Om0+Omde), wa=0)

    n_sne=len(z_s)

    
    def mu_func(z):

        if wCDM:
            d__ = wcosmo.FlatwCDM(H0, Om0, w).comoving_distance(z)
            mu=5*jnp.log10(d__*(1+z)*1e6/10)
        else:
            d__=background.transverse_comoving_distance(cosmo_jax, 1/(1+z))
            mu=5*jnp.log10((1+z)/h*d__*1e6/10)
            mu = mu[0]
    
        return mu
    
    
    mu_grad=jax.grad(mu_func)
    
    def mu_vmap(z):
    
    
        return jax.vmap(mu_func,in_axes=(0))(z)
    
    def mu_grad_vmap(z):
    
    
        return jax.vmap(mu_grad ,in_axes=(0))(z)
    
    def R_calc():
        z = jnp.array(1089.0)
        d = wcosmo.FlatwCDM(1., Om0, w).comoving_distance(z)
        d=d/(1+z)
        return jnp.sqrt(Om0)*(1+z)*d/299792.458

    R_value = R_calc()   
    if cmb_bool:
        numpyro.sample("cmb_obs",dist.Normal(jnp.array([R_value]),sigma_Rcmb), obs=jnp.array([R_cmb_obs]))


    


    with numpyro.plate("plate_i",n_sne):
        d_s=background.transverse_comoving_distance(cosmo_jax, 1/(1+z_s))
    
        mu_s=mu_vmap(z_s)
        
        mu_err= jnp.absolute(mu_grad_vmap(z_s)*(sigma_pec/299792.458))
            
        sigma_int_ = jnp.sqrt(sigma_int**2+mu_err**2)
        
        x0 = jnp.repeat(x0,n_sne)
    
    
        c0 = jnp.repeat(c0,n_sne)
        
        if model=='analytical':
            numpyro.sample("obs", SkewNormalPlus3D(mu_s+M0,c0,x0,sigma_int_,sigma_c,sigma_x,
                       alpha,beta,alpha_c,data_err_s[:,0],data_err_s[:,1],data_err_s[:,2],data_err_s[:,3],data_err_s[:,4],data_err_s[:,5],m_cut,sigma_cut,a,b),sample_shape=(1,), obs=data_s)
        elif model=='naive':
            numpyro.sample("obs", Naive(mu_s+M0,c0,x0,sigma_int_,sigma_c,sigma_x,
                       alpha,beta,alpha_c,data_err_s[:,0],data_err_s[:,1],data_err_s[:,2],data_err_s[:,3],data_err_s[:,4],data_err_s[:,5],m_cut,sigma_cut,a,b),sample_shape=(1,), obs=data_s)

        elif model=='flow':
            numpyro.sample("obs", FlowSNP3D(mu_s+M0,c0,x0,sigma_int_,jnp.repeat(sigma_c,n_sne),jnp.repeat(sigma_x,n_sne),
                       jnp.repeat(alpha,n_sne),jnp.repeat(beta,n_sne),jnp.repeat(alpha_c,n_sne),data_err_s[:,0],data_err_s[:,1],data_err_s[:,2],data_err_s[:,3],data_err_s[:,4],data_err_s[:,5],m_cut,sigma_cut,a,b),sample_shape=(1,), obs=data_s)





n =11500



seed=rep_
print(seed)


rng_key = jax.random.PRNGKey(seed)

rng_key, rng_key_ = random.split(rng_key)
z_s_=sample_redshifts(n,rng_key,Om0=0.315,w0=-1,beta_rate=1.5)


rng_key, rng_key_ = random.split(rng_key)
prior_predictive = Predictive(sample_model, num_samples=1)

prior_predictions = prior_predictive(rng_key_,z_s_,dust=False)
rng_key, rng_key_ = random.split(rng_key)

z_pec =random.normal(rng_key,(n,))*(300/299792.458)
        
z_s = (1 +z_s_)/(1+z_pec)-1

sel_sim = prior_predictions["sel_s"][0,:]


sel_sim = np.logical_and(sel_sim==1,np.logical_and(z_s>0.05,z_s<1.1)).astype(float)

d_hat_sim = prior_predictions["d_hat_s"][0,:,:]

d_s = d_hat_sim[sel_sim==1,:]

d_err_s= jnp.array([jnp.exp(prior_predictions["log_mag_err_s"][0,:][sel_sim==1.])**2,jnp.exp(prior_predictions["log_c_err_s"][0,:][sel_sim==1.])**2,jnp.exp(prior_predictions["log_x_err_s"][0,:][sel_sim==1.])**2,
                    prior_predictions["cov_m_c"][0,:][sel_sim==1.],prior_predictions["cov_m_x"][0,:][sel_sim==1.],prior_predictions["cov_c_x"][0,:][sel_sim==1.]]).T


z_s=z_s[sel_sim==1.] 

print(len(sel_sim[sel_sim==1]))


d_hat_sim=0
sel_sim =0
z_pec =0
u=0
z_s_=0
prior_predictions=0

from numpyro.infer import init_to_value

init_dict={'w':-1.1,'Om0':0.35,'Omde':0.65,'M0':-19.35,'c0_int':-0.04,'x0':-0.4,'sigma_int':0.1,'sigma_c':0.05,'sigma_x':1,'alpha':-0.11,'beta':3.0,'alpha_c':-0.01}

nuts_kernel = NUTS(mcmc_model,adapt_step_size=True,max_tree_depth=7,init_strategy=init_to_value(values=init_dict))                                                                                                   

mcmc = MCMC(nuts_kernel, num_samples=500, num_warmup=500,num_chains=4)
rng_key = random.PRNGKey(0)

mcmc.run(rng_key, z_s,data_s= d_s,data_err_s= d_err_s,wCDM=wCDM_bool,model=model_type)
mcmc.print_summary()
posterior_samples = mcmc.get_samples()

if cmb_bool:
    name+='_cmb'
dir__ = 'toy_chains_'+name
os.makedirs(dir__, exist_ok=True)

if wCDM_bool:

    np.savez(dir__+'/w'+str(rep_)+'.npz',**posterior_samples)
else:
    np.savez(dir__+'/l'+str(rep_)+'.npz',**posterior_samples)
