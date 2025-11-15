#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:08:42 2025

@author: parrenif
"""

for i in range(len(self.iceintervals_depthtop)):
    for j in range(len(self.iceintervals_depthtop)):
        if i == j: 
            self.iceintervals_correlation[i, j] = 1.
        else:
            self.iceintervals_correlation[i, j] = 0.9