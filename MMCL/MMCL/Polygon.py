#! /bin/python3

"""
Polygon.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.
"""

from HyperSpace 		import Point, Vector, Plane

import numpy as np
import sys

class Polygon(object):
	"""
	Polygon Object Class

	https://mathworld.wolfram.com/RegularPolygon.html
	"""
	def __init__(self, n, a = 1, origin = None):
		self._n = n
		self._a = a
		
		if not isinstance(origin, Point):
			if (isinstance(origin, np.ndarray) or isinstance(origin, list)):
				self._origin = Point(*origin)
			elif origin == None:
				self._origin = Point()
			else:
				raise ValueError("Origin must be Point Ojbect or list or None")
		else:
			self._origin = origin

		# exterior angle radians
		self._alpha = (n - 2) * np.pi

		# interior angle radians
		self._beta = np.pi - (self._alpha / n)

		# inradius
		self._r = a/2 * np.arctan(np.pi / n)

		# circumradius
		self._R = a/2 * np.arcsin(np.pi / n)

		# area
		self._A = n/4 * a**2 * np.arctan(np.pi / n)

	def __call__(self):
		return self._points()

	def _points(self):
		point = [
			Point(	self._origin.x + self._a * np.cos(self._beta * i), 
					self._origin.y + self._a * np.sin(self._beta * i),
					self._origin.z
			) for i in range(self._n)
		]

		# point.append(point[0])

		return point
		
