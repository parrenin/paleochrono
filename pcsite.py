#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:56:06 2019
Module for the Site class.
@author: parrenif
"""

import os
import sys
import warnings
import math as m
import numpy as np
import matplotlib.pyplot as mpl
import pandas as pd
from scipy.linalg import lu_factor, lu_solve
from numpy.linalg import cholesky
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from scipy import stats
import pickle
import yaml
from pcmath import interp_lin_aver, interp_stair_aver, grid, truncation,\
    stretch, interp_lin_slope, mean_distri, stdev_distri
import pccfg
# from numpy import interp
from numpy.core.multiarray import interp
# Should be safe, the only difference is when dealing with complex numbers
# or with a periodic function. Gain is 20s/300s on AICC2023-LowRes on xps13
import scipy.stats as stats
# from threadpoolctl import threadpool_limits

if pccfg.is_jax: 
    from jax.numpy import dot
else:
    from numpy import dot


# dummy use of the interp1d function
Fooooooo = interp1d


class Site(object):
    """This is the class for a site."""

    def __init__(self, dlab):
        self.label = dlab

        # Default parameters
        self.archive = 'icecore'
        self.deporate_prior_rep = 'staircase'
        self.age_top = None
        self.depth = np.empty(0)
        self.corr_a_age = None
        self.calc_a = False
        self.calc_a_method = None
        self.gamma_source = None
        self.beta_source = None
        self.calc_tau = False
        self.thickness = None
        self.calc_lid = False
        self.lid_value = None
        self.start = 'prior'
        self.corr_lid_age = None
        self.corr_tau_depth = None
        self.lambda_thinning = None
        self.accu0 = None
        self.beta = None
        self.pprime = None
        self.muprime = None
        self.sliding = None
        self.dens_firn = None
        self.depth_unit = 'm'
        self.age_label = 'ice'
        self.age2_label = 'air'
        self.tuning = {}
        self.restart_file = "restart.bin"
        self.fig_age_show_unc = True   #Whether to show the age uncertainty in the age figure
        self.fig_age_switch_axes = False
        self.fig_max_age_unc = None
        self.c14_cal = 'intcal20'
        self.background_correlation_deporate = 0.

# Setting of the parameters from the parameter files

        yamls = ''
        filename = pccfg.datadir+'/parameters_all_sites.yml'
        if os.path.isfile(filename):
            yamls = open(filename).read()
        filename = pccfg.datadir+self.label+'/parameters.yml'
        if os.path.isfile(filename):
            yamls += '\n'+open(filename).read()
        if yamls != '':
            para = yaml.load(yamls, Loader=yaml.FullLoader)
            self.__dict__.update(para)
            self.depth = grid(self.depth_grid)
            self.corr_deporate_age = grid(self.corr_deporate_grid)
            try:
                inf = self.age_truncation['inf']
                sup = self.age_truncation['sup']
                self.corr_deporate_age = truncation(self.corr_deporate_age, inf, sup)
            except AttributeError:
                pass
            if self.archive == 'icecore':
                self.corr_lid_age =grid(self.corr_lid_grid)
                try:
                    inf = self.age_truncation['inf']
                    sup = self.age_truncation['sup']
                    self.corr_lid_age = truncation(self.corr_lid_age, inf, sup)
                except AttributeError:
                    pass
                self.corr_thinning_depth = grid(self.corr_thinning_grid)
                self.corr_thinning_depth = stretch(self.corr_thinning_depth, self.depth[0], 
                                                   self.depth[-1])

##Translation of deprecated names

        try:
            self.calc_lid = self.calc_LID
            print('WARNING: calc_LID is deprecated. Use calc_lid instead.')
        except AttributeError:
            pass
        try:
            self.corr_lid_age = self.corr_LID_age
            print('WARNING: corr_LID_age is deprecated. Use corr_lid_age instead.')
        except AttributeError:
            pass
        try:
            self.dens_firn = self.Dfirn
            print('WARNING: Dfirn is deprecated. Use dens_firn instead.')
        except AttributeError:
            pass
        try:
            self.sliding = self.s
            print('WARNING: s is deprecated. Use sliding instead.')
        except AttributeError:
            pass
        try:
            self.accu0 = self.A0
            print('WARNING: A0 is deprecated. Use deporate0 instead.')
        except AttributeError:
            pass
        try:
            self.accu0 = self.deporate0
        except AttributeError:
            pass
        try:
            self.calc_a = self.calc_deporate
        except AttributeError:
            print('WARNING: calc_a is deprecated. Use calc_deporate instead.')
        try:
            self.calc_a_method = self.calc_deporate_method
        except AttributeError:
            print('WARNING: calc_a_method is deprecated. Use calc_deporate_method instead.')
        try:
            self.corr_a_age = self.corr_deporate_age
        except AttributeError:
            print('WARNING: corr_a_age is deprecated. Use corr_deporate_age instead.')
        try:
            self.lambda_lid = self.lambda_LID
            print('WARNING: lambda_LID is deprecated. Use lambda_lid instead.')
        except AttributeError:
            pass
        try:
            self.lambda_a = self.lambda_deporate
        except AttributeError:
            print('WARNING: lambda_a is deprecated. Use lambda_deporate instead.')
        try:
            self.lambda_tau = self.lambda_thinning
        except AttributeError:
            print('WARNING: lambda_tau is deprecated. Use lambda_thinning instead.')
        try:
            self.calc_tau = self.calc_thinning
        except AttributeError:
            print('WARNING: calc_tau is deprecated. Use calc_thinning instead.')
        if self.archive == 'icecore':
            try:
                self.corr_tau_depth = self.corr_thinning_depth
            except AttributeError:
                print('WARNING: corr_tau_depth is deprecated. Use corr_thinning_depth instead.')
        try:
            self.accu_prior_rep = self.deporate_prior_rep
        except AttributeError:
            print('WARNING: accu_prior_rep is deprecated. Use deporate_prior_rep instead.')
        try:
            self.sigmap_corr_a
            print('WARNING: sigmap_corr_a is deprecated. Use deporate_prior_sigma instead.')
        except AttributeError:
            try:
                self.sigmap_corr_a = self.deporate_prior_sigma
            except AttributeError:
                pass
        try:
            self.sigmap_corr_tau
            print('WARNING: sigmap_corr_tau is deprecated. Use thinning_prior_sigma instead.')
        except AttributeError:
            try:
                self.sigmap_corr_tau = self.thinning_prior_sigma
            except AttributeError:
                pass
        try:
            self.lid_prior_sigma = self.sigmap_corr_LID
            print('WARNING: sigmap_corr_LID is deprecated. Use lid_prior_sigma instead.')
        except AttributeError:
            pass
        try:
            self.sigmap_corr_lid
            print('WARNING: sigmap_corr_lid is deprecated. Use lid_prior_sigma instead.')
        except AttributeError:
            try:
                self.sigmap_corr_lid = self.lid_prior_sigma
            except AttributeError:
                pass
        try:
            self.age_top_prior
        except AttributeError:
            self.age_top_prior = self.age_top
            print('WARNING: Now, age_top is a variable to be optimized.'
                  'Therefore you need to define age_top_prior.'
                  'Using age_top for now.')
        try:
            self.age_top_sigma
        except AttributeError:
            self.age_top_sigma = 10.
            print('WARNING: Now, age_top is a variable to be optimized.'
                  'Therefore you need to define age_top_sigma.'
                  'Setting age_top_sigma to', self.age_top_sigma, 'for now.')
        try:
            self.lid_value = self.LID_value
            print('WARNING: Now use lid_value instead of LID_value')
        except AttributeError:
            pass
        
        if len(self.age_label)>0:
            self.age_label_ = self.age_label + '_'
            self.age_labelsp = self.age_label + ' '
        else:
            self.age_label_ = self.age_label
            self.age_labelsp = self.age_label
        if len(self.age2_label)>0:
            self.age2_label_ = self.age2_label + '_'
            self.age2_labelsp = self.age2_label + ' '
        else:
            self.age2_label_ = self.age2_label
            self.age2_labelsp = self.age2_label
        
        
        ##Initialisation of variables
        self.depth_mid = (self.depth[1:]+self.depth[:-1])/2
        self.depth_inter = (self.depth[1:]-self.depth[:-1])
        self.lid = np.empty_like(self.depth)
        self.sigma_delta_depth = np.empty_like(self.depth)
        self.sigma_airlayerthick = np.zeros_like(self.depth_mid)
        self.airlayerthick_init = np.empty_like(self.depth_mid)
        self.age_init = np.empty_like(self.depth)
        self.sigma_accu_model = np.empty_like(self.depth_mid)
        self.tau_init = np.empty_like(self.depth_mid)
        self.a_init = np.empty_like(self.depth_mid)
        self.airage_init = np.empty_like(self.depth_mid)
        self.sigma_icelayerthick = np.empty_like(self.depth_mid)
        self.airlayerthick = np.empty_like(self.depth_mid)
        self.ice_equiv_depth = np.empty_like(self.depth)
        self.sigma_tau = np.empty_like(self.depth_mid)
        self.icelayerthick = np.empty_like(self.depth_mid)
        self.icelayerthick_init = np.empty_like(self.depth_mid)
        self.sigma_tau_model = np.empty_like(self.depth_mid)
        self.delta_depth_init = np.empty_like(self.depth)
        self.sigma_lid_model = np.empty_like(self.depth)
        self.lid_init = np.empty_like(self.depth)
        self.sigma_age = np.zeros_like(self.depth)
        self.sigma_airage = np.zeros_like(self.depth)
        self.sigma_lid = np.zeros_like(self.depth)
        self.sigma_delta_age = np.zeros_like(self.depth)
        self.sigma_delta_depth = np.zeros_like(self.depth)
        self.sigma_icelayerthick = np.zeros_like(self.depth_mid)
        self.sigma_accu = np.zeros_like(self.depth_mid)
        self.sigma_tau = np.zeros_like(self.depth_mid)
        self.ulidie = np.empty_like(self.depth)
        self.cov = np.array([])

## We set up the raw model
        if self.calc_a:
            if self.archive == 'icecore':
                if self.calc_a_method == 'constant':
                    print('Prior accumulation rate is taken as constant')
                else:
                    filename = pccfg.datadir+self.label+'/isotopes.txt'
                    df = pd.read_csv(filename, sep=None, comment='#', engine='python')
                    self.iso_depth = df['depth'].to_numpy(dtype=float)
                    if self.calc_a_method == 'fullcorr':
                        self.iso_d18o_ice = df['d18O'].to_numpy(dtype=float)
                        self.d18o_ice = interp_stair_aver(self.depth, self.iso_depth, self.iso_d18o_ice)
                        self.iso_deutice = df['deut'].to_numpy(dtype=float)
                        self.deutice = interp_stair_aver(self.depth, self.iso_depth, self.iso_deutice)
                        self.iso_d18o_sw = df['d18Osw'].to_numpy(dtype=float)
                        self.d18o_sw = interp_stair_aver(self.depth, self.iso_depth, self.iso_d18o_sw)
                        self.excess = self.deutice-8*self.d18o_ice   # dans Uemura : d=excess
                        self.accu = np.empty_like(self.deutice)
                        self.d18o_ice_corr = self.d18o_ice-self.d18o_sw*(1+self.d18o_ice/1000)/\
                            (1+self.d18o_sw/1000)	#Uemura (1)
                        self.deutice_corr = self.deutice-8*self.d18o_sw*(1+self.deutice/1000)/\
                            (1+8*self.d18o_sw/1000) #Uemura et al. (CP, 2012) (2)
                        self.excess_corr = self.deutice_corr-8*self.d18o_ice_corr
                        self.deutice_fullcorr = self.deutice_corr+self.gamma_source/self.beta_source*\
                            self.excess_corr
                    elif self.calc_a_method == 'deut':
                        self.iso_deutice = df['deut'].to_numpy(dtype=float)
                        self.deutice_fullcorr = interp_stair_aver(self.depth, self.iso_depth,
                                                                  self.iso_deutice)
                    elif self.calc_a_method == 'd18O':
                        self.iso_d18o_ice = df['d18O'].to_numpy(dtype=float)
                        self.deutice_fullcorr = 8*interp_stair_aver(self.depth, self.iso_depth,
                                                                    self.iso_d18o_ice)
                    else:
                        print('Accumulation method not recognized')
                        sys.exit()
        else:
            filename = pccfg.datadir+self.label+'/deposition.txt'
            df = pd.read_csv(filename, sep=None, comment='#', engine='python')
            self.a_depth = df['depth'].to_numpy(dtype=float)
            self.a_a = df['deporate'].to_numpy(dtype=float)
            try:
                self.a_sigma = df['rel_unc'].to_numpy(dtype=float)
            except:
                pass
            if self.accu_prior_rep == 'staircase':
                self.a_model = interp_stair_aver(self.depth, self.a_depth, self.a_a)
            elif self.accu_prior_rep == 'linear':
                self.a_model = interp_lin_aver(self.depth, self.a_depth, self.a_a)
            else:
                print('Representation of prior accu scenario not recognized')
            self.accu = self.a_model


        self.age = np.empty_like(self.depth)
        self.airage = np.empty_like(self.depth)

        if self.archive == 'icecore':
            try:
                filename = pccfg.datadir+self.label+'/solid_fraction.txt'
                df = pd.read_csv(filename, sep=None, comment='#', engine='python')
            except:
                filename = pccfg.datadir+self.label+'/density.txt'
                print("Warning: 'density.txt' filename is deprecated, use 'solid_fraction.txt'")
                df = pd.read_csv(filename, sep=None, comment='#', engine='python')
            self.dens_depth = df['depth'].to_numpy(dtype=float)
            self.dens_dens = df['rel_dens'].to_numpy(dtype=float)
            #FIXME: implement staircase reprensentation for the density, as is done for accu.
            self.dens = interp(self.depth_mid, self.dens_depth, self.dens_dens)
        else:
            filename = pccfg.datadir+self.label+'/compaction_factor.txt'
            try:
                df = pd.read_csv(filename, sep=None, comment='#', engine='python')
                self.dens_depth = df['depth'].to_numpy(dtype=float)
                self.dens_dens = df['factor'].to_numpy(dtype=float)
                self.dens = interp(self.depth_mid, self.dens_depth, self.dens_dens)
            except:
                self.dens = np.ones_like(self.depth_mid)
            #FIXME: implement staircase reprensentation for the density, as is done for accu.

        if self.archive == 'icecore':

            if self.calc_tau:
                self.iedepth = np.cumsum(np.concatenate((np.array([self.iedepth_top]), 
                                                         self.dens*self.depth_inter)))
                self.iedepth_mid = (self.iedepth[1:]+self.iedepth[:-1])/2
                self.thickness_ie = self.thickness-self.depth[-1]+self.iedepth[-1]

            if self.calc_lid:
                if self.depth[0] < self.lid_value:
                    self.lid_depth = np.array([self.depth[0], self.lid_value-0.01, self.lid_value, self.depth[-1]])
                    self.lid_lid = np.array([np.nan, np.nan, self.lid_value, self.lid_value])
                else:
                    self.lid_depth = np.array([self.depth[0], self.depth[-1]])
                    self.lid_lid = np.array([self.lid_value, self.lid_value])
            else:
                filename = pccfg.datadir+self.label+'/lock_in_depth.txt'
                df = pd.read_csv(filename, sep=None, comment='#', engine='python')
                self.lid_depth = df['depth'].to_numpy(dtype=float)
                self.lid_lid = df['LID'].to_numpy(dtype=float)
                try:
                    self.lid_sigma = df['rel_unc'].to_numpy(dtype=float)
                except:
                    pass
            self.lid_model = interp(self.depth, self.lid_depth, self.lid_lid)

            self.delta_depth = np.empty_like(self.depth)
            self.udepth = np.empty_like(self.depth)

#        print 'depth_mid ', np.size(self.depth_mid)
#        print 'zeta ', np.size(self.zeta)
            if self.calc_tau:
                self.zeta = (self.thickness_ie-self.iedepth_mid)/self.thickness_ie
                self.tau = np.empty_like(self.depth_mid)
            else:
                filename = pccfg.datadir+self.label+'/thinning.txt'
                df = pd.read_csv(filename, sep=None, comment='#', engine='python')
                self.tau_depth = df['depth'].to_numpy(dtype=float)
                self.tau_tau = df['thinning'].to_numpy(dtype=float)
                try:
                    self.tau_sigma = df['rel_unc'].to_numpy(dtype=float)
                except:
                    pass
                self.tau_model = interp(self.depth_mid, self.tau_depth, self.tau_tau)
                self.tau = self.tau_model

        self.raw_model()


## Now we set up the correction functions

        if self.start == 'restart':
            with open(pccfg.datadir+self.label+'/'+self.restart_file, 'rb') as f:
                if self.archive == 'icecore':
                    resi_age_top, corr_a_age, corr_a, corr_lid_age, corr_lid, corr_tau_depth,\
                        corr_tau = pickle.load(f)
                    self.corr_lid = interp(self.corr_lid_age, corr_lid_age, corr_lid)
                    self.corr_tau = interp(self.corr_tau_depth, corr_tau_depth, corr_tau)
                    
                else:
                    resi_age_top, corr_a_age, corr_a = pickle.load(f)
            self.resi_age_top = resi_age_top
            self.corr_a = interp(self.corr_a_age, corr_a_age, corr_a)

        elif self.start == 'default' or self.start == 'prior':
            self.resi_age_top = np.array([0.])
            self.corr_a = np.zeros(np.size(self.corr_a_age))
            if self.archive == 'icecore':
                self.corr_lid = np.zeros(np.size(self.corr_lid_age))
                self.corr_tau = np.zeros(np.size(self.corr_tau_depth))
        elif self.start == 'random':
            self.resi_age_top = np.random.normal(loc=0., scale=1., size=1)
            self.corr_a = np.random.normal(loc=0., scale=1., size=np.size(self.corr_a_age))
            if self.archive == 'icecore':
                self.corr_lid = np.random.normal(loc=0., scale=1., size=np.size(self.corr_lid_age))
                self.corr_tau = np.random.normal(loc=0., scale=1.,
                                                 size=np.size(self.corr_tau_depth))
        else:
            print('Start option not recognized.')



## Definition of the covariance matrix of the background

        try:
          x_out = (self.corr_a_age[:-1]+self.corr_a_age[1:])/2
          x_out = np.concatenate((np.array([self.corr_a_age[0]]), x_out,
                                 np.array([self.corr_a_age[-1]])))
          self.sigmap_corr_a = interp_stair_aver(x_out, self.fct_age_model(self.a_depth),
                                                        self.a_sigma)
        except AttributeError:
            self.sigmap_corr_a=self.sigmap_corr_a*np.ones(np.size(self.corr_a_age))

        if self.archive == 'icecore':
            try:
                x_out = (self.corr_lid_age[:-1]+self.corr_lid_age[1:])/2
                x_out = np.concatenate((np.array([self.corr_lid_age[0]]), x_out,
                                        np.array([self.corr_lid_age[-1]])))

                lid_age = self.fct_airage_model(self.lid_depth)
                self.sigmap_corr_lid = interp_lin_aver(x_out,
                                                 lid_age[~np.isnan(lid_age)],
                                                 self.lid_sigma[~np.isnan(lid_age)])
            except AttributeError:
                self.sigmap_corr_lid=self.sigmap_corr_lid*np.ones(np.size(self.corr_lid_age))

            try:
                x_out = (self.corr_tau_depth[:-1]+self.corr_tau_depth[1:])/2
                x_out = np.concatenate((np.array([self.corr_tau_depth[0]]), x_out,
                                 np.array([self.corr_tau_depth[-1]])))
                self.sigmap_corr_tau = interp_lin_aver(x_out, self.tau_depth,
                                                 self.tau_sigma)
            except AttributeError:
                self.iedepth = np.cumsum(np.concatenate((np.array([self.iedepth_top]), 
                                                         self.dens*self.depth_inter)))
                self.thickness_ie = self.thickness-self.depth[-1]+self.iedepth[-1]
                self.sigmap_corr_tau=self.k/self.thickness_ie*interp(self.corr_tau_depth,
                                                                        self.depth, self.iedepth)

        #Accu correlation matrix
        self.correlation_corr_a = interp(np.abs(np.ones((np.size(self.corr_a_age),\
            np.size(self.corr_a_age)))*self.corr_a_age-\
            np.transpose(np.ones((np.size(self.corr_a_age),\
            np.size(self.corr_a_age)))*self.corr_a_age)),\
            np.array([0,self.lambda_a]),np.array([1, self.background_correlation_deporate]))
        
        
        if self.archive == 'icecore':
            #LID correlation matrix
            self.correlation_corr_lid = interp(np.abs(np.ones((np.size(self.corr_lid_age),\
                np.size(self.corr_lid_age)))*self.corr_lid_age-np.transpose(np.ones((np.size(\
                self.corr_lid_age),np.size(self.corr_lid_age)))*self.corr_lid_age)),\
                np.array([0,self.lambda_lid]),np.array([1, 0]))
            
            #Thinning correlation matrix
            self.correlation_corr_tau = interp(np.abs(np.ones((np.size(self.corr_tau_depth),\
                np.size(self.corr_tau_depth)))*self.corr_tau_depth-np.transpose(np.ones((np.size(\
                self.corr_tau_depth),np.size(self.corr_tau_depth)))*self.corr_tau_depth)),\
                np.array([0,self.lambda_tau]),np.array([1, 0]) )
    

        self.chol_a = cholesky(self.correlation_corr_a)
#        self.chol_a_lu_piv = lu_factor(self.chol_a) #FIXME: do we always need to do this?
        self.corr_a_jacmat = np.zeros((len(self.corr_a), len(self.age_model)-1))
        for i in range(len(self.corr_a)):
            corr_vec = self.chol_a[:, i]*self.sigmap_corr_a
            self.corr_a_jacmat[i, :] = np.interp((self.age_model[:-1]+self.age_model[1:])/2,
                                      self.corr_a_age, corr_vec)
        if self.archive == 'icecore':
            self.chol_tau = cholesky(self.correlation_corr_tau)
            self.corr_tau_jacmat = np.zeros((len(self.corr_tau), len(self.depth_mid)))
            for i in range(len(self.corr_tau)):
                corr_vec = self.chol_tau[:, i]*self.sigmap_corr_tau
                self.corr_tau_jacmat[i, :] = np.interp(self.depth_mid, self.corr_tau_depth, corr_vec)
            self.chol_lid = cholesky(self.correlation_corr_lid)
            self.corr_lid_jacmat = np.zeros((len(self.corr_lid), len(self.airage_model)))
            for i in range(len(self.corr_lid)):
                corr_vec = self.chol_lid[:, i]*self.sigmap_corr_lid
                self.corr_lid_jacmat[i, :] = np.interp(self.airage_model,
                                          self.corr_lid_age, corr_vec)

#Definition of the variable vector
            
        if self.archive == 'icecore':
            self.variables = np.concatenate((self.resi_age_top,
                                             self.corr_a, self.corr_tau, self.corr_lid))
        else:
            self.variables = np.concatenate((self.resi_age_top, self.corr_a))

#Reading of observations

        if self.archive == 'icecore':
            filename = pccfg.datadir+self.label+'/ice_age_horizons.txt'
        else:
            filename = pccfg.datadir+self.label+'/age_horizons.txt'
        if os.path.isfile(filename):
            df = pd.read_csv(filename, sep=None, comment='#', engine='python')
            self.icehorizons_depth = df['depth'].to_numpy(dtype=float)
            self.icehorizons_age = df['age'].to_numpy(dtype=float)
            self.icehorizons_sigma = df['age_unc'].to_numpy(dtype=float)
        else:
            self.icehorizons_depth = np.array([])
            self.icehorizons_age = np.array([])
            self.icehorizons_sigma = np.array([])
        self.icehorizons_type = np.empty(len(self.icehorizons_depth), dtype=object)
        self.icehorizons_type[:] = ''

        if self.archive == 'icecore':
            filename = pccfg.datadir+self.label+'/ice_age_horizons_C14.txt'
        else:
            filename = pccfg.datadir+self.label+'/age_horizons_C14.txt'
        if os.path.isfile(filename):
            df = pd.read_csv(filename, sep=None, comment='#', engine='python')
            try: 
                df['calib'] = df['calib'].mask(df['calib']=='default', self.c14_cal)
            except:
                df['calib'] = pd.Series(self.c14_cal, index=range(len(df)))
            try:
                df['age'] -= df['res_age']
                df['age_unc'] = (df['age_unc']**2+df['res_unc']**2).pow(0.5)
            except:
                pass
            print('Calibrating C14 ages')
            from iosacal import R
            for i in range(len(df)):
#                    print(df['depth'][i], df['age'][i], df['age_unc'][i], df['calib'][i])
                if pccfg.age_unit == 'yr':
                    r = R(df['age'][i], df['age_unc'][i], 'name')
                elif pccfg.age_unit == 'kyr':
                    r = R(df['age'][i]*1000, df['age_unc'][i]*1000, 'name')
                else:
                    print('Age unit should be yr or kyr')
                    sys.exit()
                cal_r = r.calibrate(df['calib'][i])
                age = mean_distri(cal_r)
                sigma = stdev_distri(cal_r)
                if pccfg.age_unit_ref == 'CE':
                    age = 1950 - age
                elif pccfg.age_unit == 'BCE':
                    age = age - 1950
                elif pccfg.age_unit_ref == 'b2k':
                    age = age + 50
                elif pccfg.age_unit_ref in ['BP', 'B1950']:
                    pass
                else:
                    print('Age reference should be CE, BCE, BP, B1950 or b2k')
                    sys.exit()
                if pccfg.age_unit == 'kyr':
                    age = age / 1000. 
                    sigma = sigma / 1000.
                elif pccfg.age_unit == 'yr':
                    pass
                else:
                    print('Age unit should be yr or kyr')
                    sys.exit()
                self.icehorizons_depth = np.append(self.icehorizons_depth, df['depth'][i])
                self.icehorizons_age = np.append(self.icehorizons_age, age)
                self.icehorizons_sigma = np.append(self.icehorizons_sigma, sigma)
                self.icehorizons_type = np.append(self.icehorizons_type, 'C14')

        if self.archive == 'icecore':
            filename = pccfg.datadir+self.label+'/ice_age_intervals.txt'
        else:
            filename = pccfg.datadir+self.label+'/age_intervals.txt'
        if os.path.isfile(filename):
            df = pd.read_csv(filename, sep=None, comment='#', engine='python')
            self.iceintervals_depthtop = df['depth_top'].to_numpy(dtype=float)
            self.iceintervals_depthbot = df['depth_bot'].to_numpy(dtype=float)
            self.iceintervals_duration = df['duration'].to_numpy(dtype=float)
            self.iceintervals_sigma = df['dur_unc'].to_numpy(dtype=float)
        else:
            self.iceintervals_depthtop = np.array([])
            self.iceintervals_depthbot = np.array([])
            self.iceintervals_duration = np.array([])
            self.iceintervals_sigma = np.array([])

        if self.archive == 'icecore':
            filename = pccfg.datadir+self.label+'/air_age_horizons.txt'
            if os.path.isfile(filename):
                df = pd.read_csv(filename, sep=None, comment='#', engine='python')
                self.airhorizons_depth = df['depth'].to_numpy(dtype=float)
                self.airhorizons_age = df['age'].to_numpy(dtype=float)
                self.airhorizons_sigma = df['age_unc'].to_numpy(dtype=float)
            else:
                self.airhorizons_depth = np.array([])
                self.airhorizons_age = np.array([])
                self.airhorizons_sigma = np.array([])

            filename = pccfg.datadir+self.label+'/air_age_intervals.txt'
            if os.path.isfile(filename):
                df = pd.read_csv(filename, sep=None, comment='#', engine='python')
                self.airintervals_depthtop = df['depth_top'].to_numpy(dtype=float)
                self.airintervals_depthbot = df['depth_bot'].to_numpy(dtype=float)
                self.airintervals_duration = df['duration'].to_numpy(dtype=float)
                self.airintervals_sigma = df['dur_unc'].to_numpy(dtype=float)
            else:
                self.airintervals_depthtop = np.array([])
                self.airintervals_depthbot = np.array([])
                self.airintervals_duration = np.array([])
                self.airintervals_sigma = np.array([])

            filename = pccfg.datadir+self.label+'/delta_depths.txt'
            if os.path.isfile(filename):
                df = pd.read_csv(filename, sep=None, comment='#', engine='python')
                self.delta_depth_depth = df['air_depth'].to_numpy(dtype=float)
                self.delta_depth_delta_depth = df['Ddepth'].to_numpy(dtype=float)
                self.delta_depth_sigma = df['Ddepth_unc'].to_numpy(dtype=float)
            else:
                self.delta_depth_depth = np.array([])
                self.delta_depth_delta_depth = np.array([])
                self.delta_depth_sigma = np.array([])


        self.icehorizons_correlation = np.diag(np.ones(np.size(self.icehorizons_depth)))
        self.iceintervals_correlation = np.diag(np.ones(np.size(self.iceintervals_depthtop)))
        if self.archive == 'icecore':
            self.airhorizons_correlation = np.diag(np.ones(np.size(self.airhorizons_depth)))
            self.airintervals_correlation = np.diag(np.ones(np.size(self.airintervals_depthtop)))
            self.delta_depth_correlation = np.diag(np.ones(np.size(self.delta_depth_depth)))
#        print self.icehorizons_correlation

        filename = pccfg.datadir+'/parameters_covariance_observations_all_sites.py'
        if os.path.isfile(filename):
            exec(open(filename).read())

        filename = pccfg.datadir+self.label+'/parameters_covariance_observations.py'
        if os.path.isfile(filename):
            exec(open(filename).read())
        
        if np.any(self.icehorizons_correlation != \
                  np.diag(np.ones(np.size(self.icehorizons_depth)))):
            self.icehorizons_correlation_bool = True
            self.icehorizons_chol = cholesky(self.icehorizons_correlation)
            #FIXME: we LU factor a triangular matrix. This is suboptimal.
            #We should set lu_piv directly instead.
            self.icehorizons_lu_piv = lu_factor(np.transpose(self.icehorizons_chol))
        else:
            self.icehorizons_correlation_bool = False
            
        if np.any(self.iceintervals_correlation != \
                  np.diag(np.ones(np.size(self.iceintervals_depthtop)))):
            self.iceintervals_correlation_bool = True
            self.iceintervals_chol = cholesky(self.iceintervals_correlation)
            self.iceintervals_lu_piv = lu_factor(np.transpose(self.iceintervals_chol))
        else:
            self.iceintervals_correlation_bool = False
            
        if self.archive == 'icecore':

            if np.any(self.airhorizons_correlation != \
                      np.diag(np.ones(np.size(self.airhorizons_depth)))):
                self.airhorizons_correlation_bool = True
                self.airhorizons_chol = cholesky(self.airhorizons_correlation)
                self.airhorizons_lu_piv = lu_factor(np.transpose(self.airhorizons_chol))
            else:
                self.airhorizons_correlation_bool = False

            if np.any(self.airintervals_correlation != \
                      np.diag(np.ones(np.size(self.airintervals_depthtop)))):
                self.airintervals_correlation_bool = True
                self.airintervals_chol = cholesky(self.airintervals_correlation)
                self.airintervals_lu_piv = lu_factor(np.transpose(self.airintervals_chol))
            else:
                self.airintervals_correlation_bool = False
                
            if np.any(self.delta_depth_correlation != \
                      np.diag(np.ones(np.size(self.delta_depth_depth)))):
                self.delta_depth_correlation_bool = True
                self.delta_depth_chol = cholesky(self.delta_depth_correlation)
                self.delta_depth_lu_piv = lu_factor(np.transpose(self.delta_depth_chol))
            else:
                self.delta_depth_correlation_bool = False

        for key in self.tuning:
            filename = pccfg.datadir+self.label+'/'+self.tuning[key]["data_file"]
            df = pd.read_csv(filename, sep=None, comment='#', engine='python')
            self.tuning[key]["data_depth"] = df['depth'].to_numpy(dtype=float)
            self.tuning[key]["data_value"] = df['data'].to_numpy(dtype=float)
            filename = pccfg.datadir+self.label+'/'+self.tuning[key]["target_file"]
            df = pd.read_csv(filename, sep=None, comment='#', engine='python')
            self.tuning[key]["target_age"] = df['age'].to_numpy(dtype=float)
            self.tuning[key]["target_value"] = df['value'].to_numpy(dtype=float)

    def raw_model(self):
        """Calculate the raw model, that is before applying correction functions."""

        self.age_top = self.age_top_prior
        
        if self.archive == 'icecore':
            #Accumulation
            if self.calc_a:
                if self.calc_a_method != 'constant':
                    self.a_model = self.accu0*np.exp(self.beta*(self.deutice_fullcorr-\
                        self.deutice_fullcorr[0])) #Parrenin et al. (CP, 2007a) 2.3 (6)
                else:
                    self.a_model = self.accu0*np.ones(np.size(self.depth_inter))
                    
            #Thinning
            if self.calc_tau:
                self.p_def = -1+m.exp(self.pprime)
                self.mu_melt = m.exp(self.muprime)
    #            self.sliding=m.tanh(self.sprime)
                #Parrenin et al. (CP, 2007a) 2.2 (3)
                omega_def = 1-(self.p_def+2)/(self.p_def+1)*(1-self.zeta)+\
                          1/(self.p_def+1)*(1-self.zeta)**(self.p_def+2)
                #Parrenin et al. (CP, 2007a) 2.2 (2)
                omega = self.sliding*self.zeta+(1-self.sliding)*omega_def
                self.tau_model = (1-self.mu_melt)*omega+self.mu_melt

            #udepth
            self.udepth_model = self.depth[0]+np.cumsum(np.concatenate((np.array([0]),\
                                self.dens/self.tau_model*self.depth_inter)))

            self.ulidie_model = self.lid_model*self.dens_firn

        else:
            if self.calc_a:
                self.a_model = self.accu0*np.ones(np.size(self.depth_inter))

        #Ice age
        if self.archive == 'icecore':
            self.icelayerthick_model = self.tau_model*self.a_model/self.dens
        else:
            self.icelayerthick_model = self.a_model/self.dens
        self.age_model = self.age_top+pccfg.way*np.cumsum(np.concatenate((np.array([0]),\
                         self.depth_inter/self.icelayerthick_model)))

        #air age
        if self.archive == 'icecore':
            self.ice_equiv_depth_model = interp(self.udepth_model-self.ulidie_model,
                                                   self.udepth_model, self.depth, left=np.nan)
            self.delta_depth_model = self.depth-self.ice_equiv_depth_model
            self.airage_model = interp(self.ice_equiv_depth_model, self.depth, self.age_model,
                                          left=np.nan, right=np.nan)
            self.airagedens_model = np.diff(self.airage_model)
            with np.errstate(divide='ignore'):
                self.airlayerthick_model = 1/self.airagedens_model

    
    def corrected_jacobian(self, full=False):
        """Calculate the Jacobian"""

        if self.archive == 'icecore':
            self.age_jac = np.zeros((1+len(self.corr_a)+len(self.corr_tau)+len(self.corr_lid), len(self.age)))
            self.airage_jac = np.zeros((1+len(self.corr_a)+len(self.corr_tau)+len(self.corr_lid), len(self.airage)))
            self.airage_jac[0, :] = self.age_top_sigma * np.ones(len(self.airage))
            self.delta_depth_jac = np.zeros((1+len(self.corr_a)+len(self.corr_tau)+len(self.corr_lid), len(self.depth)))
    #        delta_depth_jac[0, :] = np.zeros(len(depth))
            if full:
                self.accu_jac = np.zeros((1+len(self.corr_a)+len(self.corr_tau)+len(self.corr_lid), len(self.accu)))
                self.tau_jac = np.zeros((1+len(self.corr_a)+len(self.corr_tau)+len(self.corr_lid), len(self.tau)))
                self.lid_jac = np.zeros((1+len(self.corr_a)+len(self.corr_tau)+len(self.corr_lid), len(self.lid)))
                self.icelayerthick_jac = np.zeros((1+len(self.corr_a)+len(self.corr_tau)+len(self.corr_lid), len(self.icelayerthick)))
                self.agedens_jac = np.zeros((1+len(self.corr_a)+len(self.corr_tau)+len(self.corr_lid), len(self.agedens)))
                self.airagedens_jac = np.zeros((1+len(self.corr_a)+len(self.corr_tau)+len(self.corr_lid), len(self.airagedens)))
            #        icelayerthick_jac[0, :] = np.zeros(len(icelayerthick))
        else:
            self.age_jac = np.zeros((1+len(self.corr_a), len(self.age)))
            if full:
                self.accu_jac = np.zeros((1+len(self.corr_a), len(self.accu)))
                self.agedens_jac = np.zeros((1+len(self.corr_a), len(self.agedens)))

        self.age_jac[0, :] = self.age_top_sigma * np.ones(len(self.age))
            
        for i in range(len(self.corr_a)):

            agedens_vec = - self.corr_a_jacmat[i, :] * self.agedens
            if full:
                accu_vec =  self.corr_a_jacmat[i, :] * self.accu
                self.accu_jac[1+i, :] = accu_vec
                self.agedens_jac[1+i, :] = agedens_vec

        #Ice age
            age_vec = pccfg.way*np.cumsum(np.concatenate((np.array([0]), self.depth_inter*agedens_vec)))
            self.age_jac[1+i, :] = age_vec

        #Air age
            if self.archive == 'icecore':
                airage_vec = interp(self.ice_equiv_depth, self.depth, age_vec)
                self.airage_jac[1+i, :] = airage_vec
                if full:
                    icelayerthick_vec = self.corr_a_jacmat[i, :] * self.icelayerthick
                    self.icelayerthick_jac[1+i, :] = icelayerthick_vec
                    self.airagedens_jac[1+i, :] = pccfg.way*np.diff(airage_vec)

        if self.archive == 'icecore':
            
            for i in range(len(self.corr_tau)):

                agedens_vec = -self.corr_tau_jacmat[i, :] * self.agedens                
                age_vec = pccfg.way*np.cumsum(np.concatenate((np.array([0]), self.depth_inter*agedens_vec)))
                self.age_jac[1+len(self.corr_a)+i, :] = age_vec
                
                thin_vec = -self.corr_tau_jacmat[i, :] * self.dens/self.tau
                udepth_vec = np.cumsum(np.concatenate((np.array([0]), self.depth_inter*thin_vec)))
                delta_depth_vec = - interp(self.ice_equiv_depth, self.depth_mid,
                                              self.tau/self.dens) * (udepth_vec - \
                                            interp(self.ice_equiv_depth, self.depth, udepth_vec))
                airage_vec = interp(self.ice_equiv_depth, self.depth, age_vec) \
                                - pccfg.way* interp(self.ice_equiv_depth, self.depth_mid, self.agedens) * \
                                delta_depth_vec
                self.airage_jac[1+len(self.corr_a)+i, :] = airage_vec
                self.delta_depth_jac[1+len(self.corr_a)+i, :] = delta_depth_vec
                if full:
                    icelayerthick_vec = self.corr_tau_jacmat[i, :] * self.icelayerthick
                    self.icelayerthick_jac[1+len(self.corr_a)+i, :] = icelayerthick_vec
                    tau_vec = self.corr_tau_jacmat[i, :] * self.tau
                    self.tau_jac[1+len(self.corr_a)+i, :] = tau_vec
                    self.agedens_jac[1+len(self.corr_a)+i, :] = agedens_vec
                    self.airagedens_jac[1+len(self.corr_a)+i, :] = pccfg.way*np.diff(airage_vec)

                #To be continued...                

            for i in range(len(self.corr_lid)):

                lid_vec = self.corr_lid_jacmat[i, :] * self.lid
                delta_depth_vec = self.dens_firn * lid_vec * \
                                    interp(self.ice_equiv_depth, self.depth_mid, 
                                              self.tau/self.dens)
                airage_vec = - pccfg.way*interp(self.ice_equiv_depth, self.depth_mid, 
                                          self.agedens) * delta_depth_vec
                self.airage_jac[1+len(self.corr_a)+len(self.corr_tau)+i, :] = airage_vec
                self.delta_depth_jac[1+len(self.corr_a)+len(self.corr_tau)+i, :] = delta_depth_vec
                if full:
                    self.lid_jac[1+len(self.corr_a)+len(self.corr_tau)+i, :] = lid_vec
                    self.airagedens_jac[1+len(self.corr_a)+len(self.corr_tau)+i, :] = pccfg.way*np.diff(airage_vec)

    def corrected_jacobian_free(self):
        self.accu_jac = None
        self.age_jac = None
        self.airage_jac = None
        self.delta_depth_jac = None
        self.icelayerthick_jac = None
        self.tau_jac = None
        self.lid_jac = None

    def model_delta(self, var):
        """Calculate the Jacobian operator applied to a vector var."""
        
        
        age_top_delta = var[0] * self.age_top_sigma
        corr_delta = dot(self.chol_a, var[1:])*self.sigmap_corr_a
        agedens_delta = -interp((self.age_model[:-1]+self.age_model[1:])/2,
                                   self.corr_a_age, corr_delta) / self.accu
        self.age_delta = age_top_delta+pccfg.way*np.cumsum(np.concatenate((np.array([0]), self.depth_inter*\
                             agedens_delta)))

        if self.archive == 'icecore':
            print('Analytical Jacobian operator is not yet implemented for ice core archives.'
                  'Please use semi_analytical instead.')
            sys.exit()
            
    def model_adj(self, var):
        """Calculate the adjoint operator applied to a vector var."""
#        adj0 = self.age_top_sigma * np.sum(var)
        """to be continued..."""
        
        

    def corrected_model(self):
        """Calculate the age model, taking into account the correction functions."""

        #Age top
        self.age_top = self.age_top_prior + self.resi_age_top[0] * self.age_top_sigma

        #Accu
        corr = dot(self.chol_a, self.corr_a)*self.sigmap_corr_a
        self.accu = self.a_model*np.exp(interp((self.age_model[:-1]+self.age_model[1:])/2,
                                                  self.corr_a_age, corr))

        #Thinning and LID
        if self.archive == 'icecore':
            self.tau = self.tau_model*np.exp(interp(self.depth_mid, self.corr_tau_depth,\
                        dot(self.chol_tau, self.corr_tau)*self.sigmap_corr_tau))
            self.udepth = self.depth[0]+np.cumsum(np.concatenate((np.array([0]),\
                          self.dens/self.tau*self.depth_inter)))
            corr = dot(self.chol_lid, self.corr_lid)*self.sigmap_corr_lid
            self.lid = self.lid_model*np.exp(interp(self.airage_model, self.corr_lid_age, corr))
            self.ulidie = self.lid*self.dens_firn
   
        #Ice age
        if self.archive == 'icecore':
            self.icelayerthick = self.tau*self.accu/self.dens
        else:
            self.icelayerthick = self.accu/self.dens
        self.agedens = 1/self.icelayerthick
        self.age = self.age_top+pccfg.way*np.cumsum(np.concatenate((np.array([0]),\
                    self.depth_inter*self.agedens)))

        #Air age
        if self.archive == 'icecore':
            self.ice_equiv_depth = interp(self.udepth-self.ulidie, self.udepth, self.depth,
                                             left=np.nan)
            self.delta_depth = self.depth-self.ice_equiv_depth
            self.airage = interp(self.ice_equiv_depth, self.depth, self.age, left=np.nan,
                                    right=np.nan)
            self.airagedens = np.diff(self.airage)
            with np.errstate(divide='ignore'):
                self.airlayerthick = 1/self.airagedens

    def model(self, var):
        """Calculate the model from the vector var containing its variables."""

#        if self.calc_a==True:
#            self.accu0=var[index]
#            self.beta=var[index+1]
#            index=index+2
#        if self.calc_tau==True:
##            self.p_def=-1+m.exp(var[index])
##            self.sliding=var[index+1]
##            self.mu_melt=var[index+2]
##            index=index+3
#            self.pprime=var[index]
#            self.muprime=var[index+1]
#            index=index+2
        index = 0
#        self.resi_age_top = var[0:1]
        self.resi_age_top = var[0:1]
        index = 1
        self.corr_a = var[index:index+np.size(self.corr_a)]
        if self.archive == 'icecore':
            self.corr_tau = var[index+np.size(self.corr_a):\
                              index+np.size(self.corr_a)+np.size(self.corr_tau)]
            self.corr_lid = var[index+np.size(self.corr_tau)+np.size(self.corr_a):\
                        index+np.size(self.corr_tau)+np.size(self.corr_a)+np.size(self.corr_lid)]

        ##Corrected model
        self.corrected_model()

        if self.archive == 'icecore':
            return np.concatenate((self.age, self.accu, self.icelayerthick, self.airage,
                                   self.delta_depth, self.tau, self.lid, self.airagedens, 
                                   self.airlayerthick, self.age-self.airage))
        else:
            return np.concatenate((self.age, self.accu))


    def write_init(self):
        """Write the initial values of the variables in the corresponding *_init variables."""
        self.age_init = self.age
        self.a_init = self.accu
        self.icelayerthick_init = self.icelayerthick
        if self.archive == 'icecore':
            self.lid_init = self.lid
            self.tau_init = self.tau
            self.airagedens_init = self.airagedens
            self.airlayerthick_init = self.airlayerthick
            self.airage_init = self.airage
            self.delta_depth_init = self.delta_depth

    def fct_age(self, depth):
        """Return the age at given depths."""
        return interp(depth, self.depth, self.age)

    def fct_age_jac(self, depth):
        jac = []
        for i in range(len(self.variables)):
            jac.append(np.array([interp(depth, self.depth, self.age_jac[i,])]))
        return np.concatenate(jac)

    def fct_airage_jac(self, depth):
        jac = []
        for i in range(len(self.variables)):
            jac.append(np.array([interp(depth, self.depth, self.airage_jac[i,])]))
        return np.concatenate(jac)
    
    def fct_delta_depth_jac(self, depth):
        jac = []
        for i in range(len(self.variables)):
            jac.append(np.array([interp(depth, self.depth, self.delta_depth_jac[i,])]))
        return np.concatenate(jac)
    
    def fct_age_delta(self, depth):
        return interp(depth, self.depth, self.age_delta)

    def fct_age_init(self, depth):
        """Return the initial age at given depths."""
        return interp(depth, self.depth, self.age_init)

    def fct_age_model(self, depth):
        """Return the raw modelled age at given depths."""
        return interp(depth, self.depth, self.age_model)

    def fct_airage(self, depth):
        """Return the air age at given depths."""
        return interp(depth, self.depth, self.airage)

    def fct_airage_init(self, depth):
        """Return the initial air age at given depth."""
        return interp(depth, self.depth, self.airage_init)

    def fct_airage_model(self, depth):
        """Return the raw modelled air age at given depths."""
        return interp(depth, self.depth, self.airage_model)

    def fct_delta_depth(self, depth):
        """Return the delta_depth at given detphs."""
        return interp(depth, self.depth, self.delta_depth)


    def residuals(self):
        """Calculate the residuals from the vector of the variables"""
        resi_age = (self.fct_age(self.icehorizons_depth)-self.icehorizons_age)\
                   /self.icehorizons_sigma
        if self.icehorizons_correlation_bool:
            resi_age = lu_solve(self.icehorizons_lu_piv, resi_age)
        resi_iceint = (pccfg.way*(self.fct_age(self.iceintervals_depthbot)-\
                      self.fct_age(self.iceintervals_depthtop))-\
                      self.iceintervals_duration)/self.iceintervals_sigma
        if self.iceintervals_correlation_bool:
            resi_iceint = lu_solve(self.iceintervals_lu_piv, resi_iceint)
            

        if self.archive == 'icecore':
            resi_airage = (self.fct_airage(self.airhorizons_depth)-self.airhorizons_age)/\
                          self.airhorizons_sigma
            if self.airhorizons_correlation_bool:
                resi_airage = lu_solve(self.airhorizons_lu_piv, resi_airage)
            resi_airint = (pccfg.way*(self.fct_airage(self.airintervals_depthbot)-\
                           self.fct_airage(self.airintervals_depthtop))-\
                           self.airintervals_duration)/self.airintervals_sigma
            if self.airintervals_correlation_bool:
                resi_airint = lu_solve(self.airintervals_lu_piv, resi_airint)
            resi_delta_depth = (self.fct_delta_depth(self.delta_depth_depth)-\
                                self.delta_depth_delta_depth)/self.delta_depth_sigma
            if self.delta_depth_correlation_bool:
                resi_delta_depth = lu_solve(self.delta_depth_lu_piv, resi_delta_depth)
            resi = np.concatenate((resi_age, resi_airage,
                                   resi_iceint, resi_airint, resi_delta_depth))
        else:
            resi = np.concatenate((resi_age, resi_iceint))
            
        for key in self.tuning:
            if self.tuning[key]["air_proxy"]:
                data_age = self.fct_airage(self.tuning[key]["data_depth"])
            else:
                data_age = self.fct_age(self.tuning[key]["data_depth"])
            self.tuning[key]["data_age"] = data_age
            target_interp_value = interp(data_age, self.tuning[key]["target_age"], self.tuning[key]["target_value"])
            self.tuning[key]["target_interp_value"] = target_interp_value
            target_data_value = target_interp_value * self.tuning[key]["slope"] + self.tuning[key]["offset"]
            self.tuning[key]["target_data_value"] = target_data_value
            resi_tuning = (target_data_value - self.tuning[key]["data_value"]) / self.tuning[key]["sigma"]
            resi = np.concatenate((resi, resi_tuning))
            
        return resi

    def residuals_jacobian(self):
        resi_age_jac = self.fct_age_jac(self.icehorizons_depth)/self.icehorizons_sigma
        if self.icehorizons_correlation_bool:
            resi_age_jac = lu_solve(self.icehorizons_lu_piv, np.transpose(resi_age_jac))
        resi_iceint_jac = pccfg.way*(self.fct_age_jac(self.iceintervals_depthbot)-\
                      self.fct_age_jac(self.iceintervals_depthtop))/self.iceintervals_sigma
        if self.iceintervals_correlation_bool:
            resi_iceint_jac = lu_solve(self.iceintervals_lu_piv, np.transpose(resi_iceint_jac))
            resi_iceint_jac = np.transpose(resi_iceint_jac)
        if self.archive == 'icecore':
            resi_airage_jac = self.fct_airage_jac(self.airhorizons_depth)/self.airhorizons_sigma
            if self.airhorizons_correlation_bool:
                resi_airage_jac = lu_solve(self.airhorizons_lu_piv, np.transpose(resi_airage_jac))
                resi_airage_jac = np.transpose(resi_airage_jac)
            resi_airint_jac = pccfg.way*(self.fct_airage_jac(self.airintervals_depthbot)-\
                          self.fct_airage_jac(self.airintervals_depthtop))/self.airintervals_sigma
            if self.airintervals_correlation_bool:
                resi_airint_jac = lu_solve(self.airintervals_lu_piv, np.trasnpose(resi_airint_jac))
                resi_airint_jac = np.transpose(resi_airint_jac)
            resi_delta_depth_jac = self.fct_delta_depth_jac(self.delta_depth_depth)/ \
                                    self.delta_depth_sigma
            if self.delta_depth_correlation_bool:
                resi_delta_depth_jac = lu_solve(self.delta_depth_lu_piv, np.transpose(resi_delta_depth_jac))
                resi_delta_depth_jac = np.transpose(resi_delta_depth_jac)
            resi_jac = np.concatenate((resi_age_jac, resi_airage_jac, resi_iceint_jac, resi_airint_jac, 
                               resi_delta_depth_jac),
                              axis = 1)
        else:
            resi_jac = np.concatenate((resi_age_jac, resi_iceint_jac), axis=1)

        for key in self.tuning:
            # FIXME: maybe we could save that instead of re-calculatiing it.
            data_age = self.tuning[key]["data_age"]
            if self.tuning[key]["air_proxy"]:
                resi_tuning_jac = interp_lin_slope(data_age, self.tuning[key]["target_age"], self.tuning[key]["target_value"]) *\
                    self.fct_airage_jac(self.tuning[key]["data_depth"]) * self.tuning[key]["slope"] / self.tuning[key]["sigma"]
            else:
                resi_tuning_jac = interp_lin_slope(data_age, self.tuning[key]["target_age"], self.tuning[key]["target_value"]) *\
                    self.fct_age_jac(self.tuning[key]["data_depth"]) * self.tuning[key]["slope"] / self.tuning[key]["sigma"]
            resi_jac = np.concatenate((resi_jac, resi_tuning_jac), axis = 1)
            
        return resi_jac
    
    def residuals_delta(self):
        resi_age = self.fct_age_delta(self.icehorizons_depth)/self.icehorizons_sigma
        resi_iceint = (self.fct_age_delta(self.iceintervals_depthbot)-\
            self.fct_age_delta(self.iceintervals_depthtop))/self.iceintervals_sigma
        return np.concatenate((resi_age, resi_iceint))


    def cost_function(self):
        """Calculate the cost function."""
        cost = dot(self.residuals, np.transpose(self.residuals))
        return cost

    def jacobian(self):
        """Calculate the jacobian."""
        epsilon = np.sqrt(np.diag(self.cov))/100000000.
        model0 = self.model(self.variables)
        jacob = np.empty((np.size(model0), np.size(self.variables)))
        for i in np.arange(np.size(self.variables)):
            var = self.variables+0
            var[i] = var[i]+epsilon[i]
            model1 = self.model(var)
            with np.errstate(invalid='ignore'):
                jacob[:, i] = (model1-model0)/epsilon[i]
        model0 = self.model(self.variables)
        return jacob

# FIXME: Is this really needed anymore?
    def optimisation(self):
        """Optimize a site."""
        self.variables, self.cov = leastsq(self.residuals, self.variables, full_output=1)
        print(self.variables)
        print(self.cov)
        return self.variables, self.cov

    def sigma(self):
        """Calculate the error of various variables."""

        if pccfg.jacobian == 'automatic' or pccfg.jacobian == 'numerical' or \
        pccfg.jacobian == 'semi_analytical':
    
            jacob = self.jacobian()
    #        input('After calculating per site Jacobian. Program paused.')
    
            index = 0
            c_model = dot(jacob[index:index+np.size(self.age), :], dot(self.cov,\
                                   np.transpose(jacob[index:index+np.size(self.age), :])))
            self.sigma_age = np.sqrt(np.diag(c_model))
            index = index+np.size(self.age)
    #        input('After calculating sigma_age. Program paused.')
            c_model = dot(jacob[index:index+np.size(self.accu), :], dot(self.cov,\
                                   np.transpose(jacob[index:index+np.size(self.accu), :])))
            self.sigma_accu = np.sqrt(np.diag(c_model))
            self.sigma_agedens = self.sigma_accu / self.accu**2
            index = index+np.size(self.accu)
    
            self.sigma_accu_model = interp((self.age_model[1:]+self.age_model[:-1])/2,
                                              self.corr_a_age, self.sigmap_corr_a)
    
            if self.archive == 'icecore':
                c_model = dot(jacob[index:index+np.size(self.icelayerthick), :], dot(self.cov,\
                                       np.transpose(jacob[index:index+np.size(self.icelayerthick), :])))
                self.sigma_icelayerthick = np.sqrt(np.diag(c_model))
                self.sigma_agedens = - self.sigma_icelayerthick / self.icelayerthick**2
                index = index+np.size(self.icelayerthick)
                c_model = dot(jacob[index:index+np.size(self.airage), :], dot(self.cov,\
                                       np.transpose(jacob[index:index+np.size(self.airage), :])))
                self.sigma_airage = np.sqrt(np.diag(c_model))
                index = index+np.size(self.airage)
                c_model = dot(jacob[index:index+np.size(self.delta_depth), :], dot(self.cov,\
                                       np.transpose(jacob[index:index+np.size(self.delta_depth), :])))
                self.sigma_delta_depth = np.sqrt(np.diag(c_model))
                index = index+np.size(self.delta_depth)
                c_model = dot(jacob[index:index+np.size(self.tau), :], dot(self.cov,\
                                       np.transpose(jacob[index:index+np.size(self.tau), :])))
                self.sigma_tau = np.sqrt(np.diag(c_model))
                index = index+np.size(self.tau)
                c_model = dot(jacob[index:index+np.size(self.lid), :], dot(self.cov,\
                                       np.transpose(jacob[index:index+np.size(self.lid), :])))
                self.sigma_lid = np.sqrt(np.diag(c_model))
                index = index+np.size(self.lid)
                c_model = dot(jacob[index:index+np.size(self.airagedens), :], dot(self.cov,\
                                       np.transpose(jacob[index:index+np.size(self.airagedens), :])))
                self.sigma_airagedens = np.sqrt(np.diag(c_model))
                index = index+np.size(self.airagedens)
                c_model = dot(jacob[index:index+np.size(self.airlayerthick), :], dot(self.cov,\
                                       np.transpose(jacob[index:index+np.size(self.airlayerthick), :])))
                self.sigma_airlayerthick = np.sqrt(np.diag(c_model))
                index = index+np.size(self.airlayerthick)
                c_model = dot(jacob[index:index+np.size(self.age), :], dot(self.cov,\
                                       np.transpose(jacob[index:index+np.size(self.age), :])))
                self.sigma_delta_age = np.sqrt(np.diag(c_model))


        else:
            self.corrected_jacobian(full=True)
            c_model = dot(np.transpose(self.accu_jac), dot(self.cov, self.accu_jac))
            self.sigma_accu = np.sqrt(np.diag(c_model))
            c_model = dot(np.transpose(self.age_jac), dot(self.cov, self.age_jac))
            self.sigma_age = np.sqrt(np.diag(c_model))
            c_model = dot(np.transpose(self.agedens_jac), dot(self.cov, self.agedens_jac))
            self.sigma_agedens = np.sqrt(np.diag(c_model))
            
    #        input('After calculating sigma_age with the analytical method. Program paused.')
            if self.archive == 'icecore':
                c_model = dot(np.transpose(self.airage_jac), dot(self.cov, self.airage_jac))
                self.sigma_airage = np.sqrt(np.diag(c_model))
                c_model = dot(np.transpose(self.delta_depth_jac), 
                                 dot(self.cov, self.delta_depth_jac))
                self.sigma_delta_depth = np.sqrt(np.diag(c_model))
                c_model = dot(np.transpose(self.icelayerthick_jac), 
                                 dot(self.cov, self.icelayerthick_jac))
                self.sigma_icelayerthick = np.sqrt(np.diag(c_model))
                c_model = dot(np.transpose(self.tau_jac), 
                                 dot(self.cov, self.tau_jac))
                self.sigma_tau = np.sqrt(np.diag(c_model))
                c_model = dot(np.transpose(self.lid_jac), 
                                 dot(self.cov, self.lid_jac))
                self.sigma_lid = np.sqrt(np.diag(c_model))
                c_model = dot(np.transpose(self.age_jac-self.airage_jac), 
                                 dot(self.cov, self.age_jac-self.airage_jac))
                self.sigma_delta_age = np.sqrt(np.diag(c_model))
                c_model = dot(np.transpose(self.airagedens_jac), 
                                 dot(self.cov, self.airagedens_jac))
                self.sigma_airagedens = np.sqrt(np.diag(c_model))
                self.sigma_airlayerthick = self.sigma_airagedens / self.agedens**2
                

        self.sigma_accu_model = interp((self.age_model[1:]+self.age_model[:-1])/2,
                                              self.corr_a_age, self.sigmap_corr_a)
        if self.archive == 'icecore':
            self.sigma_lid_model = interp(self.age_model, self.corr_lid_age,
                                                 self.sigmap_corr_lid)
            self.sigma_tau_model = interp(self.depth_mid, self.corr_tau_depth,
                                                 self.sigmap_corr_tau)
        self.corrected_jacobian_free()

        
    def figures(self):
        """Build the figures of a site."""

        fig, ax1 = mpl.subplots()
        mpl.title(self.label+' Deposition rate')
        mpl.xlabel('Optimized age ('+pccfg.age_unit+' '+pccfg.age_unit_ref+')')
        mpl.ylabel('Deposition rate ('+self.depth_unit+'/'+pccfg.age_unit+')')
        if pccfg.show_initial:
            mpl.step(self.age, np.concatenate((self.a_init, np.array([self.a_init[-1]]))),
                     color=pccfg.color_init, where='post', label='Initial')
        mpl.step(self.age, np.concatenate((self.a_model, np.array([self.a_model[-1]]))),
                 color=pccfg.color_mod, where='post', label='Prior')
        mpl.step(self.age, np.concatenate((self.accu, np.array([self.accu[-1]]))),
                 color=pccfg.color_opt,
                 where='post', label='Posterior $\\pm\\sigma$')
        mpl.fill_between(self.age[:-1], np.exp(np.log(self.accu)-self.sigma_accu/self.accu),
                         np.exp(np.log(self.accu)+self.sigma_accu/self.accu),
                         color=pccfg.color_ci, label="Confidence interval")
        x_low, x_up, y_low, y_up = mpl.axis()
        if pccfg.time_axis:
            mpl.axis((x_low, self.age_top, y_low, y_up))
        else:
            mpl.axis((self.age_top, x_up, y_low, y_up))
        ax2 = ax1.twinx()
        ax2.plot((self.corr_a_age[1:]+self.corr_a_age[:-1])/2, 
                 self.corr_a_age[1:]-self.corr_a_age[:-1], label='resolution',
                 color=pccfg.color_resolution)
        ax2.set_ylabel('resolution ('+pccfg.age_unit+')')
        ax2.spines['right'].set_color(pccfg.color_resolution)
        ax2.yaxis.label.set_color(pccfg.color_resolution)
        ax2.tick_params(axis='y', colors=pccfg.color_resolution)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
        mpl.savefig(pccfg.datadir+self.label+'/deposition.'+pccfg.fig_format,
                    format=pccfg.fig_format, bbox_inches='tight')
        if not pccfg.show_figures:
            mpl.close()

        fig, ax1 = mpl.subplots()
        mpl.title(self.label+' Deposition rate (log)')
        mpl.xlabel('Optimized age ('+pccfg.age_unit+' '+pccfg.age_unit_ref+')')
        mpl.ylabel('Deposition rate ('+self.depth_unit+'/'+pccfg.age_unit+')')
        if pccfg.show_initial:
            mpl.step(self.age, np.concatenate((self.a_init, np.array([self.a_init[-1]]))),
                     color=pccfg.color_init, where='post', label='Initial')
        mpl.step(self.age, np.concatenate((self.a_model, np.array([self.a_model[-1]]))),
                 color=pccfg.color_mod, where='post', label='Prior')
        mpl.step(self.age, np.concatenate((self.accu, np.array([self.accu[-1]]))),
                 color=pccfg.color_opt,
                 where='post', label='Posterior $\\pm\\sigma$')
        mpl.fill_between(self.age[:-1], np.exp(np.log(self.accu)-self.sigma_accu/self.accu),
                         np.exp(np.log(self.accu)+self.sigma_accu/self.accu),
                         color=pccfg.color_ci, label="Confidence interval")
        ax1.axes.set_yscale('log')
        x_low, x_up, y_low, y_up = mpl.axis()
        if pccfg.time_axis:
            mpl.axis((x_low, self.age_top, y_low, y_up))
        else:
            mpl.axis((self.age_top, x_up, y_low, y_up))
        ax2 = ax1.twinx()
        ax2.plot((self.corr_a_age[1:]+self.corr_a_age[:-1])/2, 
                 self.corr_a_age[1:]-self.corr_a_age[:-1], label='resolution',
                 color=pccfg.color_resolution)
        ax2.set_ylabel('resolution ('+pccfg.age_unit+')')
        ax2.spines['right'].set_color(pccfg.color_resolution)
        ax2.yaxis.label.set_color(pccfg.color_resolution)
        ax2.tick_params(axis='y', colors=pccfg.color_resolution)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
        mpl.savefig(pccfg.datadir+self.label+'/deposition_log.'+pccfg.fig_format,
                    format=pccfg.fig_format, bbox_inches='tight')
        if not pccfg.show_figures:
            mpl.close()

        fig, ax1 = mpl.subplots()
        mpl.title(self.label+' '+self.age_labelsp+'age')
        mpl.xlabel('age ('+pccfg.age_unit+' '+pccfg.age_unit_ref+')')
        mpl.ylabel('depth ('+self.depth_unit+')')
        if pccfg.show_initial:
            mpl.plot(self.age_init, self.depth, color=pccfg.color_init, label='Initial')
        if np.size(self.icehorizons_depth[np.where(self.icehorizons_type=='')]) > 0:
            mpl.errorbar(self.icehorizons_age[np.where(self.icehorizons_type=='')],
                         self.icehorizons_depth[np.where(self.icehorizons_type=='')],
                         color=pccfg.color_obs,
                         xerr=self.icehorizons_sigma[np.where(self.icehorizons_type=='')], linestyle='', marker='o', markersize=2,
                         label="dated horizons")
        if np.size(self.icehorizons_depth[np.where(self.icehorizons_type=='C14')]) > 0:
            mpl.errorbar(self.icehorizons_age[np.where(self.icehorizons_type=='C14')],
                         self.icehorizons_depth[np.where(self.icehorizons_type=='C14')],
                         color=pccfg.color_c14,
                         xerr=self.icehorizons_sigma[np.where(self.icehorizons_type=='C14')],
                         linestyle='', marker='d', markersize=2,
                         label="$^{14}$C dated horizons")
        mpl.plot(self.age_model, self.depth, color=pccfg.color_mod, label='Prior')
        mpl.plot(self.age, self.depth, color=pccfg.color_opt,
                 label='Posterior $\\pm\\sigma$')
        mpl.fill_betweenx(self.depth, self.age-self.sigma_age, self.age+self.sigma_age,
                          color=pccfg.color_ci, label="Confidence interval")
        x_low, x_up, y_low, y_up = mpl.axis()
        mpl.axis((x_low, x_up, self.depth[-1], self.depth[0]))
        if self.fig_age_show_unc: 
            ax2 = ax1.twiny()
            ax2.plot(self.sigma_age, self.depth, color=pccfg.color_sigma,
                     label='1$\\sigma$')
            x_low, x_up, y_low, y_up = mpl.axis()
            if self.fig_max_age_unc is None:
                mpl.axis((0., x_up*5, y_low, y_up))
            else:
                mpl.axis((0., self.fig_max_age_unc, y_low, y_up))
            ax2.set_xlabel('1$\\sigma$ uncertainty ('+pccfg.age_unit+')')
            ax2.spines['top'].set_color(pccfg.color_sigma)
            ax2.xaxis.label.set_color(pccfg.color_sigma)
            ax2.tick_params(axis='x', colors=pccfg.color_sigma)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            mpl.legend(loc="best")
        mpl.savefig(pccfg.datadir+self.label+'/'+self.age_label_+'age.'+pccfg.fig_format,
                    format=pccfg.fig_format, bbox_inches='tight')
        if not pccfg.show_figures:
            mpl.close()

        fig, ax1 = mpl.subplots()
        mpl.title(self.label+' '+self.age_labelsp+'age')
        mpl.ylabel('age ('+pccfg.age_unit+' '+pccfg.age_unit_ref+')')
        mpl.xlabel('depth ('+self.depth_unit+')')
        if pccfg.show_initial:
            mpl.plot(self.depth, self.age_init, color=pccfg.color_init, label='Initial')
        if np.size(self.icehorizons_depth[np.where(self.icehorizons_type=='')]) > 0:
            mpl.errorbar(self.icehorizons_depth[np.where(self.icehorizons_type=='')],
                         self.icehorizons_age[np.where(self.icehorizons_type=='')],
                         color=pccfg.color_obs,
                         yerr=self.icehorizons_sigma[np.where(self.icehorizons_type=='')],
                         linestyle='', marker='o', markersize=2,
                         label="dated horizons")
        if np.size(self.icehorizons_depth[np.where(self.icehorizons_type=='C14')]) > 0:
            mpl.errorbar(self.icehorizons_depth[np.where(self.icehorizons_type=='C14')],
                         self.icehorizons_age[np.where(self.icehorizons_type=='C14')],
                         color=pccfg.color_obs,
                         yerr=self.icehorizons_sigma[np.where(self.icehorizons_type=='C14')],
                         linestyle='', marker='D', markersize=2,
                         label="C14 dated horizons")
        mpl.plot(self.depth, self.age_model, color=pccfg.color_mod, label='Prior')
        mpl.plot(self.depth, self.age, color=pccfg.color_opt,
                 label='Posterior $\\pm\\sigma$')
        mpl.fill_between(self.depth, self.age-self.sigma_age, self.age+self.sigma_age,
                          color=pccfg.color_ci, label="Confidence interval")
        x_low, x_up, y_low, y_up = mpl.axis()
        mpl.axis((self.depth[0], self.depth[-1], y_low, y_up))
        if self.fig_age_show_unc: 
            ax2 = ax1.twinx()
            ax2.plot(self.depth, self.sigma_age, color=pccfg.color_sigma,
                     label='1$\\sigma$')
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((x_low, x_up, 0, y_up*5))
            ax2.set_ylabel('1$\\sigma$ uncertainty ('+pccfg.age_unit+')')
            ax2.spines['right'].set_color(pccfg.color_sigma)
            ax2.yaxis.label.set_color(pccfg.color_sigma)
            ax2.tick_params(axis='y', colors=pccfg.color_sigma)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            mpl.legend(loc="best")
        if self.fig_age_switch_axes:
            mpl.savefig(pccfg.datadir+self.label+'/'+self.age_label_+'age_switch_axes.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
        if not pccfg.show_figures:
            mpl.close()

        fig, ax = mpl.subplots()
        mpl.title(self.label+' '+self.age_labelsp+'age density')
        mpl.xlabel('age density ('+pccfg.age_unit+'/'+self.depth_unit+')')
        mpl.ylabel('Depth ('+self.depth_unit+')')
        if pccfg.show_initial:
            mpl.plot(1/self.icelayerthick_init, self.depth_mid, color=pccfg.color_init,
                     label='Initial')
        mpl.plot(1/self.icelayerthick_model, self.depth_mid, color=pccfg.color_mod, label='Prior')
        mpl.plot(self.agedens, self.depth_mid, color=pccfg.color_opt,
                 label='Posterior $\\pm\\sigma$')
        mpl.fill_betweenx(self.depth_mid, np.exp(np.log(self.agedens)-self.sigma_agedens/self.agedens),
                          np.exp(np.log(self.agedens)+self.sigma_agedens/self.agedens), color=pccfg.color_ci,
                          label="Confidence interval")
        x_low, x_up, y_low, y_up = mpl.axis()
        mpl.axis((0, x_up, self.depth[-1], self.depth[0]))
        for i in range(np.size(self.iceintervals_duration)):
            y_low = self.iceintervals_depthtop[i]
            y_up = self.iceintervals_depthbot[i]
            x_low = self.iceintervals_duration[i]/(y_up-y_low)
            x_up = x_low
            xseries = np.array([x_low, x_up, x_up, x_low, x_low])
            yseries = np.array([y_low, y_low, y_up, y_up, y_low])
            if i == 0:
                mpl.plot(xseries, yseries, color=pccfg.color_obs, label="dated intervals")
                mpl.errorbar(x_up, (y_low+y_up)/2, color=pccfg.color_obs, xerr=self.iceintervals_sigma[i]/(y_up-y_low),
                             capsize=1)
            else:
                mpl.plot(xseries, yseries, color=pccfg.color_obs)
                mpl.errorbar(x_up, (y_low+y_up)/2, color=pccfg.color_obs, xerr=self.iceintervals_sigma[i]/(y_up-y_low),
                             capsize=1)
        mpl.legend(loc="best")
        mpl.savefig(pccfg.datadir+self.label+'/'+self.age_label_+'age_density.'+pccfg.fig_format,
                    format=pccfg.fig_format, bbox_inches='tight')
        if not pccfg.show_figures:
            mpl.close()

        fig, ax = mpl.subplots()
        mpl.title(self.label+' '+self.age_labelsp+'age density (log)')
        mpl.xlabel('age density ('+pccfg.age_unit+'/'+self.depth_unit+')')
        mpl.ylabel('Depth ('+self.depth_unit+')')
        if pccfg.show_initial:
            mpl.plot(1/self.icelayerthick_init, self.depth_mid, color=pccfg.color_init,
                     label='Initial')
        mpl.plot(1/self.icelayerthick_model, self.depth_mid, color=pccfg.color_mod, label='Prior')
        mpl.plot(self.agedens, self.depth_mid, color=pccfg.color_opt,
                 label='Posterior $\\pm\\sigma$')
        mpl.fill_betweenx(self.depth_mid, np.exp(np.log(self.agedens)-self.sigma_agedens/self.agedens),
                          np.exp(np.log(self.agedens)+self.sigma_agedens/self.agedens), color=pccfg.color_ci,
                          label="Confidence interval")
        ax.axes.set_xscale('log')
        x_low, x_up, y_low, y_up = mpl.axis()
        mpl.axis((x_low, x_up, self.depth[-1], self.depth[0]))
        for i in range(np.size(self.iceintervals_duration)):
            y_low = self.iceintervals_depthtop[i]
            y_up = self.iceintervals_depthbot[i]
            x_low = self.iceintervals_duration[i]/(y_up-y_low)
            x_up = x_low
            xseries = np.array([x_low, x_up, x_up, x_low, x_low])
            yseries = np.array([y_low, y_low, y_up, y_up, y_low])
            if i == 0:
                mpl.plot(xseries, yseries, color=pccfg.color_obs, label="dated intervals")
                mpl.errorbar(x_up, (y_low+y_up)/2, color=pccfg.color_obs, xerr=self.iceintervals_sigma[i]/(y_up-y_low),
                             capsize=1)
            else:
                mpl.plot(xseries, yseries, color=pccfg.color_obs)
                mpl.errorbar(x_up, (y_low+y_up)/2, color=pccfg.color_obs, xerr=self.iceintervals_sigma[i]/(y_up-y_low),
                             capsize=1)
        mpl.legend(loc="best")
        mpl.savefig(pccfg.datadir+self.label+'/'+self.age_label_+'age_density_log.'+pccfg.fig_format,
                    format=pccfg.fig_format, bbox_inches='tight')
        if not pccfg.show_figures:
            mpl.close()

        if self.archive == 'icecore':

            fig, ax = mpl.subplots()
            mpl.title(self.label+' '+self.age_labelsp+'layer thickness')
            mpl.xlabel('thickness of layers ('+self.depth_unit+'/'+pccfg.age_unit+')')
            mpl.ylabel('Depth ('+self.depth_unit+')')
            if pccfg.show_initial:
                mpl.plot(self.icelayerthick_init, self.depth_mid, color=pccfg.color_init,
                         label='Initial')
            mpl.plot(self.icelayerthick_model, self.depth_mid, color=pccfg.color_mod, label='Prior')
            mpl.plot(self.icelayerthick, self.depth_mid, color=pccfg.color_opt,
                     label='Posterior $\\pm\\sigma$')
            mpl.fill_betweenx(self.depth_mid, self.icelayerthick-self.sigma_icelayerthick,
                              self.icelayerthick+self.sigma_icelayerthick, color=pccfg.color_ci,
                              label="Confidence interval")
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((0, x_up, self.depth[-1], self.depth[0]))
            mpl.legend(loc="best")
            mpl.savefig(pccfg.datadir+self.label+'/'+self.age_label_+'layer_thickness.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()

            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' thinning')
            mpl.xlabel('Thinning')
            mpl.ylabel('Depth ('+self.depth_unit+')')
            if pccfg.show_initial:
                mpl.plot(self.tau_init, self.depth_mid, color=pccfg.color_init, label='Initial')
            mpl.plot(self.tau_model, self.depth_mid, color=pccfg.color_mod, label='Prior')
            mpl.plot(self.tau, self.depth_mid, color=pccfg.color_opt,
                     label='Posterior $\\pm\\sigma$')
            mpl.fill_betweenx(self.depth_mid, self.tau-self.sigma_tau, self.tau+self.sigma_tau,
                              color=pccfg.color_ci, label="Confidence interval")
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((x_low, x_up, self.depth[-1], self.depth[0]))
            ax2 = ax1.twiny()
            ax2.plot(self.corr_tau_depth[1:]-self.corr_tau_depth[:-1], 
                     (self.corr_tau_depth[1:]+self.corr_tau_depth[:-1])/2, label='resolution',
                     color=pccfg.color_resolution)
            ax2.set_xlabel('resolution ('+self.depth_unit+')')
            ax2.spines['top'].set_color(pccfg.color_resolution)
            ax2.xaxis.label.set_color(pccfg.color_resolution)
            ax2.tick_params(axis='x', colors=pccfg.color_resolution)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
            mpl.savefig(pccfg.datadir+self.label+'/thinning.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()

            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' thinning (log)')
            mpl.xlabel('Thinning')
            mpl.ylabel('Depth ('+self.depth_unit+')')
            if pccfg.show_initial:
                mpl.plot(self.tau_init, self.depth_mid, color=pccfg.color_init, label='Initial')
            mpl.plot(self.tau_model, self.depth_mid, color=pccfg.color_mod, label='Prior')
            mpl.plot(self.tau, self.depth_mid, color=pccfg.color_opt,
                     label='Posterior $\\pm\\sigma$')
            mpl.fill_betweenx(self.depth_mid, np.exp(np.log(self.tau)-self.sigma_tau/self.tau),
                              np.exp(np.log(self.tau)+self.sigma_tau/self.tau),
                              color=pccfg.color_ci, label="Confidence interval")
            ax1.axes.set_xscale('log')
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((x_low, x_up, self.depth[-1], self.depth[0]))
            ax2 = ax1.twiny()
            ax2.plot(self.corr_tau_depth[1:]-self.corr_tau_depth[:-1], 
                     (self.corr_tau_depth[1:]+self.corr_tau_depth[:-1])/2, label='resolution',
                     color=pccfg.color_resolution)
            ax2.set_xlabel('resolution ('+self.depth_unit+')')
            ax2.spines['top'].set_color(pccfg.color_resolution)
            ax2.xaxis.label.set_color(pccfg.color_resolution)
            ax2.tick_params(axis='x', colors=pccfg.color_resolution)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
            mpl.savefig(pccfg.datadir+self.label+'/thinning_log.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()

            mpl.title(self.label+' '+self.age2_labelsp+'age density')
            mpl.xlabel('age density ('+pccfg.age_unit+'/'+self.depth_unit+')')
            mpl.ylabel('Depth ('+self.depth_unit+')')
            if pccfg.show_initial:
                mpl.plot(self.airagedens_init, self.depth_mid, color=pccfg.color_init,
                         label='Initial')
            mpl.plot(self.airagedens_model, self.depth_mid, color=pccfg.color_mod,
                     label='Prior')
            mpl.plot(self.airagedens, self.depth_mid, color=pccfg.color_opt,
                     label='Posterior $\\pm\\sigma$')
            mpl.fill_betweenx(self.depth_mid, self.airagedens-self.sigma_airagedens,
                              self.airagedens+self.sigma_airagedens,
                              color=pccfg.color_ci, label="Confidence interval")
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((x_low, x_up, self.depth[-1], self.depth[0]))
            for i in range(np.size(self.airintervals_duration)):
                y_low = self.airintervals_depthtop[i]
                y_up = self.airintervals_depthbot[i]
                x_low = self.airintervals_duration[i]/(y_up-y_low)
                x_up = x_low
                xseries = np.array([x_low, x_up, x_up, x_low, x_low])
                yseries = np.array([y_low, y_low, y_up, y_up, y_low])
                if i == 0:
                    mpl.plot(xseries, yseries, color=pccfg.color_obs, label="dated intervals")
                    mpl.errorbar(x_up, (y_low+y_up)/2, color=pccfg.color_obs, xerr=self.airintervals_sigma[i]/(y_up-y_low),
                                 capsize=1)
                else:
                    mpl.plot(xseries, yseries, color=pccfg.color_obs)
                    mpl.errorbar(x_up, (y_low+y_up)/2, color=pccfg.color_obs, xerr=self.airintervals_sigma[i]/(y_up-y_low),
                                 capsize=1)
            mpl.legend(loc="best")
            mpl.savefig(pccfg.datadir+self.label+'/'+self.age2_label_+'age_density.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()

            if pccfg.show_airlayerthick:
                fig, ax = mpl.subplots()
                mpl.title(self.label+' '+self.age2_labelsp+'layer thickness')
                mpl.xlabel('thickness of annual layers ('+self.depth_unit+'/'+pccfg.age_unit+')')
                mpl.ylabel('Depth ('+self.depth_unit+')')
                if pccfg.show_initial:
                    mpl.plot(self.airlayerthick_init, self.depth_mid, color=pccfg.color_init,
                             label='Initial')
                mpl.plot(self.airlayerthick_model, self.depth_mid, color=pccfg.color_mod,
                         label='Prior')
                mpl.plot(self.airlayerthick, self.depth_mid, color=pccfg.color_opt,
                         label='Posterior $\\pm\\sigma$')
                mpl.fill_betweenx(self.depth_mid, self.airlayerthick-self.sigma_airlayerthick,
                                  self.airlayerthick+self.sigma_airlayerthick,
                                  color=pccfg.color_ci, label="Confidence interval")
                x_low, x_up, y_low, y_up = mpl.axis()
                mpl.axis((x_low, x_up, self.depth[-1], self.depth[0]))
                mpl.legend(loc="best")
                mpl.savefig(pccfg.datadir+self.label+'/'+self.age2_label_+'layer_thickness.'+pccfg.fig_format,
                            format=pccfg.fig_format, bbox_inches='tight')
                if not pccfg.show_figures:
                    mpl.close()

            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' Lock-In Depth')
            mpl.xlabel('Optimized age ('+pccfg.age_unit+' '+pccfg.age_unit_ref+')')
            mpl.ylabel('LID ('+self.depth_unit+')')
            if pccfg.show_initial:
                mpl.plot(self.airage, self.lid_init, color=pccfg.color_init, label='Initial')
            mpl.plot(self.airage, self.lid_model, color=pccfg.color_mod, label='Prior')
            mpl.plot(self.airage, self.lid, color=pccfg.color_opt,
                     label='Posterior $\\pm\\sigma$')
            mpl.fill_between(self.age, self.lid-self.sigma_lid, self.lid+self.sigma_lid,
                             color=pccfg.color_ci, label="Confidence interval")
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((self.age_top, x_up, y_low, y_up))
            ax2 = ax1.twinx()
            ax2.plot((self.corr_lid_age[1:]+self.corr_lid_age[:-1])/2, 
                     self.corr_lid_age[1:]-self.corr_lid_age[:-1], label='resolution',
                     color=pccfg.color_resolution)
            ax2.set_ylabel('resolution ('+pccfg.age_unit+')')
            ax2.spines['right'].set_color(pccfg.color_resolution)
            ax2.yaxis.label.set_color(pccfg.color_resolution)
            ax2.tick_params(axis='y', colors=pccfg.color_resolution)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
            mpl.savefig(pccfg.datadir+self.label+'/lock_in_depth.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()

            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' $\\Delta$age')
            mpl.xlabel('Optimized '+self.age2_labelsp+'age ('+pccfg.age_unit+' '+pccfg.age_unit_ref+')')
            mpl.ylabel('$\\Delta$age ('+pccfg.age_unit+')')
            if pccfg.show_initial:
                mpl.plot(self.airage, self.age_init-self.airage_init, color=pccfg.color_init,
                         label='Initial')
            mpl.plot(self.airage, self.age_model-self.airage_model, color=pccfg.color_mod,
                     label='Prior')
            mpl.plot(self.airage, self.age-self.airage, color=pccfg.color_opt,
                     label='Posterior $\\pm\\sigma$')
            mpl.fill_between(self.airage, self.age-self.airage-self.sigma_delta_age,
                             self.age-self.airage+self.sigma_delta_age,
                             color=pccfg.color_ci, label="Confidence interval")
            x_low, x_up, y_low, y_up = mpl.axis()
            if pccfg.time_axis:
                mpl.axis((x_low, self.age_top, y_low, y_up))
            else:
                mpl.axis((self.age_top, x_up, y_low, y_up))
            mpl.savefig(pccfg.datadir+self.label+'/delta_age.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()

            fig, ax1 = mpl.subplots()
#            mpl.figure(self.label+' air age')
            mpl.title(self.label+' '+self.age2_labelsp+'age')
            mpl.xlabel('age ('+pccfg.age_unit+' '+pccfg.age_unit_ref+')')
            mpl.ylabel('depth ('+self.depth_unit+')')
            if pccfg.show_initial:
                mpl.plot(self.airage_init, self.depth, color=pccfg.color_init, label='Initial')
            if np.size(self.airhorizons_depth) > 0:
                mpl.errorbar(self.airhorizons_age, self.airhorizons_depth, color=pccfg.color_obs,
                             xerr=self.airhorizons_sigma, linestyle='', marker='o', markersize=2,
                             label="observations")
            mpl.plot(self.airage_model, self.depth, color=pccfg.color_mod, label='Prior')
            mpl.fill_betweenx(self.depth, self.airage-self.sigma_airage,
                              self.airage+self.sigma_airage,
                              color=pccfg.color_ci, label="Confidence interval")
            mpl.plot(self.airage, self.depth, color=pccfg.color_opt,
                     label='Posterior $\\pm\\sigma$')
            x_low, x_up, y_low, y_up = mpl.axis()
            if pccfg.time_axis:
                mpl.axis((x_low, self.age_top, self.depth[-1], self.depth[0]))
            else:
                mpl.axis((self.age_top, x_up, self.depth[-1], self.depth[0]))
            ax2 = ax1.twiny()
            ax2.plot(self.sigma_airage, self.depth, color=pccfg.color_sigma,
                     label='1$\\sigma$')
            x_low, x_up, y_low, y_up = mpl.axis()
            if self.fig_max_age_unc is None:
                mpl.axis((0., x_up*5, y_low, y_up))
            else:
                mpl.axis((0., self.fig_max_age_unc, y_low, y_up))
            ax2.set_xlabel('1$\\sigma$ uncertainty ('+pccfg.age_unit+')')
            ax2.spines['top'].set_color(pccfg.color_sigma)
            ax2.xaxis.label.set_color(pccfg.color_sigma)
            ax2.tick_params(axis='x', colors=pccfg.color_sigma)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
            mpl.savefig(pccfg.datadir+self.label+'/'+self.age2_label_+'age.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()

            fig, ax = mpl.subplots()
            mpl.title(self.label+' $\\Delta$depth')
            mpl.xlabel('$\\Delta$depth ('+self.depth_unit+')')
            mpl.ylabel(self.age2_labelsp+'depth ('+self.depth_unit+')')
            if pccfg.show_initial:
                mpl.plot(self.delta_depth_init, self.depth, color=pccfg.color_init, label='Initial')
            if np.size(self.delta_depth_depth) > 0:
                mpl.errorbar(self.delta_depth_delta_depth, self.delta_depth_depth,
                             color=pccfg.color_obs,
                             xerr=self.delta_depth_sigma, linestyle='', marker='o', markersize=2,
                             label="observations")
            mpl.plot(self.delta_depth_model, self.depth, color=pccfg.color_mod, label='Prior')
            mpl.plot(self.delta_depth, self.depth, color=pccfg.color_opt,
                     label='Posterior $\\pm\\sigma$')
            mpl.fill_betweenx(self.depth, self.delta_depth-self.sigma_delta_depth,
                              self.delta_depth+self.sigma_delta_depth, color=pccfg.color_ci,
                              label="Confidence interval")
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((x_low, x_up, self.depth[-1], self.depth[0]))
            mpl.legend(loc='best')
            mpl.savefig(pccfg.datadir+self.label+'/delta_depth.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()

            for key in self.tuning:
                
                fig, ax1 = mpl.subplots()
                mpl.title(self.label+' '+key+' tuning')
                mpl.xlabel('age ('+pccfg.age_unit+' '+pccfg.age_unit_ref+')')
                mpl.ylabel(key+' ('+self.tuning[key]["unit"]+')')
                x = self.tuning[key]["target_age"]
                y = self.tuning[key]["target_value"] * self.tuning[key]["slope"] + self.tuning[key]["offset"]
                mpl.plot(x, y, label='target', color='r')
                if self.tuning[key]["air_proxy"]:
                    x = interp(self.tuning[key]["data_depth"], self.depth, self.airage)
                else:
                    x = interp(self.tuning[key]["data_depth"], self.depth, self.age)
                y = self.tuning[key]["data_value"]
                mpl.plot(x, y, label='record', color='k')
                mpl.legend()
                mpl.savefig(pccfg.datadir+self.label+'/'+key+'_tuning.'+pccfg.fig_format,
                            format=pccfg.fig_format, bbox_inches='tight')
                if not pccfg.show_figures:
                    mpl.close()
                    
                fig, ax1 = mpl.subplots()
                mpl.title(self.label+' '+key+' regression')
                mpl.xlabel(self.tuning[key]["target_name"]+' ('+self.tuning[key]["target_unit"]+")")
                mpl.ylabel(key+' ('+self.tuning[key]["unit"]+')')
                mpl.scatter(self.tuning[key]["target_interp_value"], self.tuning[key]["data_value"])
                b, a, r_value, p_value, std_err = stats.linregress(self.tuning[key]["target_interp_value"], self.tuning[key]["data_value"])
                xseq = np.linspace(np.min(self.tuning[key]["target_interp_value"]), np.max(self.tuning[key]["target_interp_value"]), num=10)
                r2_value = r_value**2
                std_err = m.sqrt(np.mean((self.tuning[key]["target_data_value"]-self.tuning[key]["data_value"])**2))
                mpl.plot(xseq, a+b*xseq, color='k', lw=2.,
                         label=f"y={b:.2e}x {a:+.2e}, R$^2$={r2_value:.2f}, stderr={std_err:.2e}")
                mpl.legend(framealpha=0.5)
                mpl.savefig(pccfg.datadir+self.label+'/'+key+'_regression.'+pccfg.fig_format,
                            format=pccfg.fig_format, bbox_inches='tight')
                if not pccfg.show_figures:
                    mpl.close()

        # Plotting residuals and detecting outliers
        
        if pccfg.show_prior_residuals:
            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' Prior residuals')
            mpl.xlabel('Residuals (no unit)')
            mpl.ylabel('Probability density')
            resi = self.variables
            rms = m.sqrt(np.sum(resi**2)/len(resi))
            mini = np.min(resi, initial=0)
            maxi = np.max(resi, initial=0)
            student = stats.t.fit(resi)
            mpl.hist(resi, bins=40, range=(-4., 4.), density=True, 
                     label=f"RMS: {rms:.3}, min: {mini:.3}, max: {maxi:.3},\n"
                     f"loc: {student[1]:.3}, scale: {student[2]:.3}, df: {student[0]:.3e}")
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((-4., 4., y_low, y_up))
            mpl.legend()
            mpl.savefig(pccfg.datadir+self.label+'/prior_residuals.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()

            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' Deporate residuals')
            mpl.xlabel('Residuals (no unit)')
            mpl.ylabel('Probability density')
            resi = self.corr_a
            rms = m.sqrt(np.sum(resi**2)/len(resi))
            mini = np.min(resi, initial=0)
            maxi = np.max(resi, initial=0)
            student = stats.t.fit(resi)
            mpl.hist(resi, bins=40, range=(-4., 4.), density=True, 
                     label=f"RMS: {rms:.3}, min: {mini:.3}, max: {maxi:.3},\n"
                     f"loc: {student[1]:.3}, scale: {student[2]:.3}, df: {student[0]:.3e}")
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((-4., 4., y_low, y_up))
            mpl.legend()
            mpl.savefig(pccfg.datadir+self.label+'/deporate_residuals.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()

        # corr = dot(self.chol_a, self.corr_a)*self.sigmap_corr_a
        # step = min(np.abs(self.corr_a_age[1:]-self.corr_a_age[:-1]))
        # age_reg = np.arange(min(self.corr_a_age), max(self.corr_a_age), step)
        # corr_reg = np.interp(age_reg, self.corr_a_age, corr)
        # fig, ax1 = mpl.subplots()
        # mpl.title(self.label+' Deporate autocorrelation')
        # ax1.acorr(corr_reg, usevlines=True, normed=True, maxlags=50, lw=2)
        # mpl.savefig(pccfg.datadir+self.label+'/deporate_autocorrelation.'+pccfg.fig_format,
        #             format=pccfg.fig_format, bbox_inches='tight')
        # if not pccfg.show_figures:
        #     mpl.close()

        if self.archive == 'icecore' and pccfg.show_prior_residuals:
            
            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' Thinning residuals')
            mpl.xlabel('Residuals (no unit)')
            mpl.ylabel('Probability density')
            resi = self.corr_tau
            rms = m.sqrt(np.sum(resi**2)/len(resi))
            mini = np.min(resi, initial=0)
            maxi = np.max(resi, initial=0)
            student = stats.t.fit(resi)
            mpl.hist(resi, bins=40, range=(-4., 4.), density=True, 
                     label=f"RMS: {rms:.3}, min: {mini:.3}, max: {maxi:.3},\n"
                     f"loc: {student[1]:.3}, scale: {student[2]:.3}, df: {student[0]:.3e}")
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((-4., 4., y_low, y_up))
            mpl.legend()
            mpl.savefig(pccfg.datadir+self.label+'/thinning_residuals.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()
                
            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' Lock-in depth residuals')
            mpl.xlabel('Residuals (no unit)')
            mpl.ylabel('Probability density')
            resi = self.corr_lid
            rms = m.sqrt(np.sum(resi**2)/len(resi))
            mini = np.min(resi, initial=0)
            maxi = np.max(resi, initial=0)
            student = stats.t.fit(resi)
            mpl.hist(resi, bins=40, range=(-4., 4.), density=True, 
                     label=f"RMS: {rms:.3}, min: {mini:.3}, max: {maxi:.3},\n"
                     f"loc: {student[1]:.3}, scale: {student[2]:.3}, df: {student[0]:.3e}")
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((-4., 4., y_low, y_up))
            mpl.legend()
            mpl.savefig(pccfg.datadir+self.label+'/lock_in_depth_residuals.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()


        resi = self.residuals()
        if np.size(resi)>0:
            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' Observations residuals')
            mpl.xlabel('Residuals (no unit)')
            mpl.ylabel('Probability density')
            rms = m.sqrt(np.sum(resi**2)/len(resi))
            mini = np.min(resi, initial=0)
            maxi = np.max(resi, initial=0)
            student = stats.t.fit(resi)
            mpl.hist(resi, bins=40, range=(-4., 4.), density=True, 
                     label=f"RMS: {rms:.3}, min: {mini:.3}, max: {maxi:.3},\n"
                     f"loc: {student[1]:.3}, scale: {student[2]:.3}, df: {student[0]:.3e}")
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((-4., 4., y_low, y_up))
            mpl.legend()
            mpl.savefig(pccfg.datadir+self.label+'/obs_residuals.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()


        resi = (self.fct_age(self.icehorizons_depth)-self.icehorizons_age)\
                   /self.icehorizons_sigma
        resi_simple = (self.fct_age(self.icehorizons_depth[np.where(self.icehorizons_type=='')])-self.icehorizons_age[np.where(self.icehorizons_type=='')])\
                   /self.icehorizons_sigma[np.where(self.icehorizons_type=='')]
        resi_C14 = (self.fct_age(self.icehorizons_depth[np.where(self.icehorizons_type=='C14')])-self.icehorizons_age[np.where(self.icehorizons_type=='C14')])\
                   /self.icehorizons_sigma[np.where(self.icehorizons_type=='C14')]
        if np.size(resi)>0:
            for i in np.where(np.abs(resi_simple)>pccfg.outlier_level)[0]:
                print('Outlier in '+self.age_labelsp+'age horizon:', str(i+1)+"/"+str(len(resi_simple)), 'at depth:',
                      self.icehorizons_depth[np.where(self.icehorizons_type=='')][i], self.depth_unit+",",
                      'level:', "{:.2f}".format(np.abs(resi_simple[i])))
            for i in np.where(np.abs(resi_C14)>pccfg.outlier_level)[0]:
                print('Outlier in '+self.age_labelsp+'C14 age horizon:', str(i+1)+"/"+str(len(resi_C14)), 'at depth:',
                      self.icehorizons_depth[np.where(self.icehorizons_type=='C14')][i], self.depth_unit+",",
                      'level:', "{:.2f}".format(np.abs(resi_C14[i])))
            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' '+self.age_label+' age horizons residuals')
            mpl.xlabel('Residuals (no unit)')
            mpl.ylabel('Probability density')
            rms = m.sqrt(np.sum(resi**2)/len(resi))
            mini = np.min(resi, initial=0)
            maxi = np.max(resi, initial=0)
            student = stats.t.fit(resi)
            mpl.hist(resi, bins=40, range=(-4., 4.), density=True, 
                     label=f"RMS: {rms:.3}, min: {mini:.3}, max: {maxi:.3},\n"
                     f"loc: {student[1]:.3}, scale: {student[2]:.3}, df: {student[0]:.3e}")
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((-4., 4., y_low, y_up))
            mpl.legend()
            mpl.savefig(pccfg.datadir+self.label+'/'+self.age_label_+'age_horizons_residuals.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()

        resi = (self.fct_age(self.iceintervals_depthbot)-\
                      self.fct_age(self.iceintervals_depthtop)-\
                      self.iceintervals_duration)/self.iceintervals_sigma
        if np.size(resi)>0:
            for i in np.where(np.abs(resi)>pccfg.outlier_level)[0]:
                print('Outlier in', self.age_label,'age interval at index:', str(i+1)+"/"+str(len(resi)),
                      'at top depth:',
                      self.iceintervals_depthtop[i], self.depth_unit+", level:", "{:.2f}".format(np.abs(resi[i])))
            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' '+self.age_label+' age intervals residuals')
            mpl.xlabel('Residuals (no unit)')
            mpl.ylabel('Probability density')
            rms = m.sqrt(np.sum(resi**2)/len(resi))
            mini = np.min(resi, initial=0)
            maxi = np.max(resi, initial=0)
            student = stats.t.fit(resi)
            mpl.hist(resi, bins=40, range=(-4., 4.), density=True, 
                     label=f"RMS: {rms:.3}, min: {mini:.3}, max: {maxi:.3},\n"
                     f"loc: {student[1]:.3}, scale: {student[2]:.3}, df: {student[0]:.3e}")
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((-4., 4., y_low, y_up))
            mpl.legend()
            mpl.savefig(pccfg.datadir+self.label+'/'+self.age_label_+'age_intervals_residuals.'+pccfg.fig_format,
                        format=pccfg.fig_format, bbox_inches='tight')
            if not pccfg.show_figures:
                mpl.close()

        if self.archive == 'icecore':
            resi = (self.fct_airage(self.airhorizons_depth)-self.airhorizons_age)/\
                          self.airhorizons_sigma
            if np.size(resi)>0:
                for i in np.where(np.abs(resi)>pccfg.outlier_level)[0]:
                    print('Outlier in', self.age2_label,'age horizon:', str(i+1)+"/"+str(len(resi)),
                          'at depth:', self.airhorizons_depth[i], self.depth_unit+", level:", "{:.2f}".format(np.abs(resi[i])))
                fig, ax1 = mpl.subplots()
                mpl.title(self.label+' '+self.age2_label+' age horizons residuals')
                mpl.xlabel('Residuals (no unit)')
                mpl.ylabel('Probability density')
                rms = m.sqrt(np.sum(resi**2)/len(resi))
                mini = np.min(resi, initial=0)
                maxi = np.max(resi, initial=0)
                student = stats.t.fit(resi)
                mpl.hist(resi, bins=40, range=(-4., 4.), density=True, 
                         label=f"RMS: {rms:.3}, min: {mini:.3}, max: {maxi:.3},\n"
                         f"loc: {student[1]:.3}, scale: {student[2]:.3}, df: {student[0]:.3e}")
                x_low, x_up, y_low, y_up = mpl.axis()
                mpl.axis((-4., 4., y_low, y_up))
                mpl.legend()
                mpl.savefig(pccfg.datadir+self.label+'/'+self.age2_label_+'age_horizons_residuals.'+pccfg.fig_format,
                            format=pccfg.fig_format, bbox_inches='tight')
                if not pccfg.show_figures:
                    mpl.close()

                resi = (self.fct_airage(self.airintervals_depthbot)-\
                               self.fct_airage(self.airintervals_depthtop)-\
                               self.airintervals_duration)/self.airintervals_sigma
            if np.size(resi)>0:
                for i in np.where(np.abs(resi)>pccfg.outlier_level)[0]:
                    print('Outlier in', self.age2_label,'age interval:', str(i+1)+"/"+str(len(resi)),
                          'and top depth:', self.airintervals_depthtop[i], self.depth_unit+", level:", "{:.2f}".format(np.abs(resi[i])))
                fig, ax1 = mpl.subplots()
                mpl.title(self.label+' '+self.age2_label+' age intervals residuals')
                mpl.xlabel('Residuals (no unit)')
                mpl.ylabel('Probability density')
                rms = m.sqrt(np.sum(resi**2)/len(resi))
                mini = np.min(resi, initial=0)
                maxi = np.max(resi, initial=0)
                student = stats.t.fit(resi)
                mpl.hist(resi, bins=40, range=(-4., 4.), density=True, 
                         label=f"RMS: {rms:.3}, min: {mini:.3}, max: {maxi:.3},\n"
                         f"loc: {student[1]:.3}, scale: {student[2]:.3}, df: {student[0]:.3e}")
                x_low, x_up, y_low, y_up = mpl.axis()
                mpl.axis((-4., 4., y_low, y_up))
                mpl.legend()
                mpl.savefig(pccfg.datadir+self.label+'/'+self.age2_label_+'age_intervals_residuals.'+pccfg.fig_format,
                            format=pccfg.fig_format, bbox_inches='tight')
                if not pccfg.show_figures:
                    mpl.close()
               
            resi = (self.fct_delta_depth(self.delta_depth_depth)-\
                       self.delta_depth_delta_depth)/self.delta_depth_sigma
            if np.size(resi)>0:
                for i in np.where(np.abs(resi)>pccfg.outlier_level)[0]:
                    print('Outlier in $\\Delta$depth obs.:', str(i+1)+"/"+str(len(resi)),
                          'at air depth:', self.delta_depth_depth[i], 
                          self.depth_unit+", level:", "{:.2f}".format(np.abs(resi[i])))
                fig, ax1 = mpl.subplots()
                mpl.title(self.label+' $\\Delta$depth residuals')
                mpl.xlabel('Residuals (no unit)')
                mpl.ylabel('Probability density')
                rms = m.sqrt(np.sum(resi**2)/len(resi))
                mini = np.min(resi, initial=0)
                maxi = np.max(resi, initial=0)
                student = stats.t.fit(resi)
                mpl.hist(resi, bins=40, range=(-4., 4.), density=True, 
                         label=f"RMS: {rms:.3}, min: {mini:.3}, max: {maxi:.3},\n"
                         f"loc: {student[1]:.3}, scale: {student[2]:.3}, df: {student[0]:.3e}")
                x_low, x_up, y_low, y_up = mpl.axis()
                mpl.axis((-4., 4., y_low, y_up))
                mpl.legend()
                mpl.savefig(pccfg.datadir+self.label+'/delta_depth_residuals.'+pccfg.fig_format,
                            format=pccfg.fig_format, bbox_inches='tight')
                if not pccfg.show_figures:
                    mpl.close()
 
        for key in self.tuning:
            if self.tuning[key]["air_proxy"]:
                data_age = self.fct_airage(self.tuning[key]["data_depth"])
            else:
                data_age = self.fct_age(self.tuning[key]["data_depth"])
            self.tuning[key]["data_age"] = data_age
            target_interp_value = interp(data_age, self.tuning[key]["target_age"], self.tuning[key]["target_value"])
            self.tuning[key]["target_interp_value"] = target_interp_value
            target_data_value = target_interp_value * self.tuning[key]["slope"] + self.tuning[key]["offset"]
            self.tuning[key]["target_data_value"] = target_data_value
            resi = (target_data_value - self.tuning[key]["data_value"]) / self.tuning[key]["sigma"]
            if np.size(resi)>0:
                for i in np.where(np.abs(resi)>pccfg.outlier_level)[0]:
                    print('Outlier in', key, 'tuning obs.:', str(i+1)+"/"+str(len(resi)),
                          'and air depth:', self.tuning[key]["data_depth"][i], 
                          self.depth_unit+", level:", "{:.2f}".format(np.abs(resi[i])))
                fig, ax1 = mpl.subplots()
                mpl.title(self.label+' '+key+' tuning residuals')
                mpl.xlabel('Residuals (no unit)')
                mpl.ylabel('Probability density')
                rms = m.sqrt(np.sum(resi**2)/len(resi))
                mini = np.min(resi, initial=0)
                maxi = np.max(resi, initial=0)
                student = stats.t.fit(resi)
                mpl.hist(resi, bins=40, range=(-4., 4.), density=True, 
                         label=f"RMS: {rms:.3}, min: {mini:.3}, max: {maxi:.3},\n"
                         f"loc: {student[1]:.3}, scale: {student[2]:.3}, df: {student[0]:.3e}")
                x_low, x_up, y_low, y_up = mpl.axis()
                mpl.axis((-4., 4., y_low, y_up))
                mpl.legend()
                mpl.savefig(pccfg.datadir+self.label+'/'+key+'_tuning_residuals.'+pccfg.fig_format,
                            format=pccfg.fig_format, bbox_inches='tight')
                if not pccfg.show_figures:
                    mpl.close()


    def save(self):
        """Save various variables for a site."""
        if self.archive == 'icecore':
            output = np.vstack((self.depth, self.age, self.sigma_age, self.airage,
                                self.sigma_airage,
                                self.sigma_delta_age,
                                np.append(self.accu, self.accu[-1]),
                                np.append(self.sigma_accu, self.sigma_accu[-1]),
                                np.append(self.tau, self.tau[-1]),
                                np.append(self.sigma_tau, self.sigma_tau[-1]),
                                self.lid, self.sigma_lid,
                                self.delta_depth, self.sigma_delta_depth,
                                np.append(self.a_model, self.a_model[-1]),
                                np.append(self.sigma_accu_model, self.sigma_accu_model[-1]),
                                np.append(self.tau_model, self.tau_model[-1]),
                                np.append(self.sigma_tau_model, self.sigma_tau_model[-1]),
                                self.lid_model, self.sigma_lid_model,
                                np.append(self.icelayerthick, self.icelayerthick[-1]),
                                np.append(self.sigma_icelayerthick, self.sigma_icelayerthick[-1]),
                                np.append(self.airlayerthick, self.airlayerthick[-1]),
                                np.append(self.sigma_airlayerthick, self.sigma_airlayerthick[-1])))
        else:
            output = np.vstack((self.depth, self.age, self.sigma_age,
                                np.append(self.accu, self.accu[-1]),
                                np.append(self.sigma_accu, self.sigma_accu[-1]),
                                np.append(self.a_model, self.a_model[-1]),
                                np.append(self.sigma_accu_model, self.sigma_accu_model[-1])))
        with open(pccfg.datadir+self.label+'/output.txt', 'w') as file_save:
            if self.archive == 'icecore':
                file_save.write('#depth\t'+self.age_label_+'age\tsigma_'+self.age_label_+'age\t'
                                ''+self.age2_label_+'age\tsigma_'+self.age2_label_+'age'
                                '\tsigma_delta_age\tdeporate'
                                '\tsigma_deporate\tthinning\tsigma_thinning\tLID\tsigma_LID'
                                 '\tdelta_depth\tsigma_delta_depth\tdeporate_model'
                                 '\tsigma_deporate_model'
                                 '\tthinning_model\tsigma_thinning_model\tLID_model'
                                 '\tsigma_LID_model\t'+self.age_label+'layerthick\t'
                                 'sigma_'+self.age_label+'layerthick'
                                 '\t'+self.age2_label+'layerthick\tsigma_'+self.age2_label+
                                 'layerthick\n')
            else:
                file_save.write('#depth\t'+self.age_label+'age\tsigma_'+self.age_label+
                                'age\tdeporate\tsigma_deporate\tdeporate_model'
                                '\tsigma_deporate_model\n')
            np.savetxt(file_save, np.transpose(output), delimiter='\t')
            file_save.close()
        resi_age_top = self.resi_age_top
        corr_a_age = self.corr_a_age
        corr_a = self.corr_a
        if self.archive == 'icecore':
            corr_lid_age = self.corr_lid_age
            corr_lid = self.corr_lid
            corr_tau_depth = self.corr_tau_depth
            corr_tau = self.corr_tau
        with open(pccfg.datadir+self.label+'/restart.bin', 'wb') as f:
            if self.archive == 'icecore':
                pickle.dump([resi_age_top, corr_a_age, corr_a, corr_lid_age,
                             corr_lid, corr_tau_depth, corr_tau], f)
            else:
                pickle.dump([resi_age_top, corr_a_age, corr_a], f)
             
    def check_airage_inversion(self):
        """Check if there are air age inversion"""
        if self.archive == 'icecore':
            if (self.airlayerthick < 0).any():
                return True
        return False
