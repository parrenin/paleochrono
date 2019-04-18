#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 10:00:12 2019
Some mathematical functions for paleochrono.
@author: parrenif
"""
import numpy as np

def interp_lin_aver(x_out, x_in, y_in):
    """Return a linear interpolation of a (x_in,y_in) series at x_out abscissas with averaging."""
    y_out = np.nan*np.zeros(np.size(x_out)-1)
    if x_out[0] < min(x_in):
        x_mod = np.concatenate((np.array([x_out[0]]), x_in))
        y_mod = np.concatenate((np.array([y_in[0]]), y_in))
    else:
        x_mod = x_in+0
        y_mod = y_in+0
    if x_out[-1] > max(x_in):
        x_mod = np.concatenate((x_mod, np.array([x_out[-1]])))
        y_mod = np.concatenate((y_mod, np.array([y_in[-1]])))
    for i in range(np.size(x_out)-1):
        x_loc = x_mod[np.where(np.logical_and(x_mod > x_out[i], x_mod < x_out[i+1]))]
        x_loc = np.concatenate((np.array([x_out[i]]), x_loc, np.array([x_out[i+1]])))
        y_loc = np.interp(x_loc, x_mod, y_mod)
        y_out[i] = np.sum((y_loc[1:]+y_loc[:-1])/2*(x_loc[1:]-x_loc[:-1]))/(x_out[i+1]-x_out[i])
    return y_out

def interp_stair_aver(x_out, x_in, y_in):
    """Return a staircase interpolation of a (x_in,y_in) series at x_out abscissas with averaging.
    """
    x_mod = x_in+0
    y_mod = y_in+0
    if x_out[0] < x_in[0]:
        x_mod = np.concatenate((np.array([x_out[0]]), x_mod))
        y_mod = np.concatenate((np.array([y_in[0]]), y_mod))
    if x_out[-1] > x_in[-1]:
        x_mod = np.concatenate((x_mod, np.array([x_out[-1]])))
        y_mod = np.concatenate((y_mod, np.array([y_in[-1]])))
    y_int = np.cumsum(np.concatenate((np.array([0]), y_mod[:-1]*(x_mod[1:]-x_mod[:-1]))))
#Maybe this is suboptimal since we compute twice g(xp[i]):
    y_out = (np.interp(x_out[1:], x_mod, y_int)-np.interp(x_out[:-1], x_mod, y_int))/\
            (x_out[1:]-x_out[:-1])
    return y_out


def gaussian(x_in):
    """Return the value of the gaussian function (no multiplicative constant)
    at a given x_in abscissa."""
    return np.exp(-x_in**2/2)