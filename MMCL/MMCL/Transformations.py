#! /bin/python3

"""
Transformations.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.
"""
from MMCL.HyperSpace import Point, Vector, Plane

import numpy as np
import sys

def scale(s, dim = 3):
	"""
	Scaling factor
	 1  0  0  0
	 0  1  0  0
	 0  0  1  0
	 0  0  0  s
	"""
	ZRM = np.identity(dim + 1)
	ZRM[-1][-1] = s
	return ZRM

def reflection_XY_Matrix(dim = 3):
	"""
	Reflection about the Z axis
	 1  0  0  0
	 0  1  0  0
	 0  0 -1  0
	 0  0  0  1
	"""
	ZRM = np.identity(dim + 1)
	ZRM[dim-1][dim-1] = -1
	return ZRM

def reflection_YZ_Matrix(dim = 3):
	"""
	Reflection about the X axis
	-1  0  0  0
	 0  1  0  0
	 0  0  1  0
	 0  0  0  1
	"""
	ZRM = np.identity(dim + 1)
	ZRM[0][0] = -1
	return ZRM

def reflection_XZ_Matrix(dim = 3):
	"""
	Reflection about the Y axis
	 1  0  0  0
	 0 -1  0  0
	 0  0  1  0
	 0  0  0  1
	"""
	ZRM = np.identity(dim + 1)
	ZRM[dim-2][dim-2] = -1
	return ZRM

def translation(o_prime, o, dim = 3):
	"""
	Translation
	 1  0  0  x
	 0  1  0  y
	 0  0  1  z
	 0  0  0  1
	"""
	diff = np.array(o_prime) - np.array(o)
	TM = np.identity(dim + 1)
	TM[:dim, -1] = -diff
	return TM

def rotation(v1, v2 = None, dim = 3):
	"""
	Rotation Matrix

	rotates one matrix onto another
	"""

	if not isinstance(v1, Point):
		if (isinstance(v1, np.ndarray) or isinstance(v1, list)):
			v1 = Point(*v1)
		else:
			raise ValueError("Point must be Point Ojbect or list")
	else:
		v1 = Point(*v1.euclidean)
	
	if not isinstance(v2, Point):
		if (isinstance(v2, np.ndarray) or isinstance(v2, list)):
			v2 = Point(*v2)
		elif v2 == None:
			v2 = Point()
		else:
			raise ValueError("Point must be Point Ojbect or list or None")
	else:
		v2 = Point(*v2.euclidean)

	# normalize vector length
	v1 /= np.linalg.norm(v1)
	# get rotation axis
	uvw = Vector(v2, v1)
	# compute trig values
	rcos = np.dot(v1, v2)
	rsin = np.linalg.norm(uvw.coefs)

	# normalize
	if not np.isclose(rsin, 0):
		# uvw /= rsin
		uvw.coefs /= rsin

	coefs = np.array([uvw.coefs])

	# Compute rotation matrix
	R = np.identity(len(v1))
	R[:dim,:dim] *= rcos
	R[:dim,:dim] += rsin * uvw.skew()
	R[:dim,:dim] += (1.0 - rcos) * coefs.T * coefs
	return R





	