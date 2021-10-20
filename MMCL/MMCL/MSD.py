#! /bin/python3

"""
MSD.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.
"""

import numpy as np

def MSD(traj, t):
	"""
	Mean Squared Displacement

	traj 	= 	trajectory coordinates matrix
	t 		= 	time step
	"""
	return np.mean([np.square(np.linalg.norm(cord[t] - cord[0])) for cord in traj])

def analyticalMSD(D, t, d):
	"""
	Analytical Mean Squared Displacement

	D 		= 	diffusion coefficient
	t 		= 	time step
	d 		= 	dimesion
	"""
	return 2 * d * D * t





