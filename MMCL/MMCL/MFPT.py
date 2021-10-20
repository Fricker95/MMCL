#! /bin/python3

"""
MFPT.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.
"""

import numpy as np
import sys

def getR(D, dt):
	# ∆x² = 6D ∆t
	# ∆x = R/4
	return np.sqrt(6 * D * dt) * 4

def getD(R, dt):
	# ∆x² = 6D ∆t
	# ∆x = R/4
	return ((R / 4)**2) / (dt * 6)

def mfpt_analytic_3D(D, R1, R2):
		return (R2**2)/(3*D) * (R2/R1 - (1/2))

def mfpt_analytic_2D(D, R1, R2):
		return (R2**2)/(2*D) * (np.log(R2/R1) - (1/2))

def adjustable_dt(distance):
	return np.log(distance)/4



