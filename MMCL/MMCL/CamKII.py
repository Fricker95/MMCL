#! /bin/python3

"""
CamKII.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.
"""

from MMCL.Polygon 			import Polygon
from MMCL.HyperSpace 		import Point, Vector, Plane
from MMCL.HyperSphere 		import Intersection, HyperSphere

import numpy as np
import sys

class CamKII(object):
	"""CamKII Object Class"""

	class Subunit(HyperSphere):
		"""Subunit Object Class"""
		def __init__(self, radius = 1, dim = 3, origin = None):
			# super(Subunit, self).__init__()
			HyperSphere.__init__(self, radius, dim, origin)
			self.activated = False
			self.intersection_point = None

	def __init__(self, n = 6, a = 1, l = 2, dim = 3, origin = None):
		self._n = n
		if not isinstance(origin, Point):
			if (isinstance(origin, np.ndarray) or isinstance(origin, list)):
				self._origin = Point(*origin)
			elif origin == None:
				self._origin = Point()
			else:
				raise ValueError("Origin must be Point Ojbect or list or None")
		else:
			self._origin = origin

		if dim == 3:
			upper = Polygon(n, a = a, origin = self._origin + [0,0,l]).__call__()
			lower = Polygon(n, a = a, origin = self._origin - [0,0,l]).__call__()
			
			self._objects = np.empty((3, n), dtype = type(self.Subunit()))

			self._objects[0] = [self.Subunit(a, origin = o) for o in upper]
			self._objects[1] = [Vector(upper[i], lower[i]) for i in range(n)]
			if l > 0:
				self._objects[2] = [self.Subunit(a, origin = o) for o in lower]

			self._perimeter = HyperSphere(l + a, origin = origin)
		else:
			upper = Polygon(n, a = a, origin = self._origin + [0,0,l]).__call__()
			self._objects = np.empty((1, n), dtype = type(self.Subunit()))
			self._objects[0] = [self.Subunit(a, dim = dim, origin = o) for o in upper]
			self._perimeter = self.Subunit(l + a, dim = dim, origin = origin)

	def __iter__(self):
		for i in range(len(self._objects)):
			for j in range(self._n):
				yield self._objects[i][j]

	def line_intersection(self, v):

		if not isinstance(v, Vector):
			if isinstance(v, Point):
				v = Vector(v)
			elif (isinstance(v, np.ndarray) or isinstance(v, list)):
				v = Vector(v)
			else:
				raise ValueError("Norm must be Vector Ojbect or list")

		dist = []
		flag = False
		for j in self:
			if isinstance(j, Subunit):
				# type, intersection
				tp, inters = j.line_intersection(v)
				if tp == Intersection.segment_partial or tp == Intersection.segment_full:
					print(tp, inters)
					if tp == Intersection.segment_full:
						dist.append((v.distance(inters[0], on = True), j))
						dist.append((v.distance(inters[1], on = True), j))
					else:
						dist.append((v.distance(inters, on = True), j))
					flag = True
			elif isinstance(j, Vector):
				continue
			else:
				raise ValueError("CamKII should return Objects")

		if flag == True:
			# calculate the minimum distance of intersection to sphere
			min_d = min([d for d,i in dist])
			# store only the closest sphere
			s = [i for d,i in dist if d == min_d][0]
			return (flag, min_d, s)

		return (flag, None, None)




