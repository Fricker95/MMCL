#! /bin/python3

"""
Brownian.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.
"""

from HyperSpace 		import Point, Vector, Plane

import numpy as np

def brownian(D, dt, d = 2, origin = None):
	"""
	Multi dimensional brownian motion simulator

	D 	=  	diffusion coefficient
	T 	= 	total time
	dt 	= 	delta t
	d 	= 	dimension
	"""

	if not isinstance(origin, Point):
		if (isinstance(origin, np.ndarray) or isinstance(origin, list)):
			origin = Point(*origin)
		elif origin == None:
			origin = Point()
		else:
			raise ValueError("Origin must be Point Ojbect or list or None")
	else:
		origin = origin

	# eta matrix
	eta = np.random.randn(1, d)

	# comparison check if dimension is 1D
	comp = (d == 1)

	# matrix of zeros 
	# if d > 1: 2 x d 
	# if d = 1: 2 x (d + 1)
	coord = np.zeros((2, d + comp))

	# insert origin
	coord[0] = origin.euclidean[:d]

	# iterate over the dimension
	for dim in np.arange(d):
		# computes the X approximation for each time tick in each dimension
		coord[1][dim + comp] = coord[0][dim + comp] + np.sqrt(2 * D * dt) * eta[0][dim]
	yield Vector(coord[1], coord[0])

  