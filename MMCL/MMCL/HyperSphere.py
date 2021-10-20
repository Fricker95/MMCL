#! /bin/python3

"""
HyperSphere.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.
"""
from MMCL.Transformations 	import *

from MMCL.Utils 			import isBetween
from MMCL.HyperSpace 		import Point, Vector, Plane

import numpy as np
import enum
import sys

class Intersection(enum.Enum):
	line_intersect 		= 0
	line_no_intersect 	= 1
	segment_outside 	= 2
	segment_inside 		= 3
	segment_partial 	= 4
	segment_tangent		= 5
	segment_full 		= 6

class HyperSphere(object):
	"""
	HyperSphere Object Class
	"""

	def __init__(self, radius = 1, dim = 3, origin = None):
		self._radius 	= float(radius)
		self._dim 		= dim
		
		if not isinstance(origin, Point):
			if (isinstance(origin, np.ndarray) or isinstance(origin, list)):
				self._origin = Point(*origin)
			elif origin == None:
				self._origin = Point()
			else:
				raise ValueError("Origin must be Point Ojbect or list or None")
		else:
			self._origin = origin

		if self._dim < 1:
			raise ValueError("dimension needs to be greater >= 1")

	def __call__(self, size = 100):
		return self._points(size)

	def __iter__(self):
		for point in self._points():
			yield point

	def __getattr__(self, attr):
		if attr == "radius":
			return self._radius
		elif attr == "dim":
			return self._dim
		elif attr == "origin":
			return self._origin
		else:
			raise AttributeError(f"{self.__class__.__name__} object has no attribute {attr}")

	def __repr__(self):
		return f"R = {self._radius}, dim = {self._dim}, origin = {self._origin.homogeneous}"

	def __str__(self):
		return f"R = {self._radius}, dim = {self._dim}, origin = {self._origin.euclidean}"

	def area(self, alpha = 0):
		"""
		a = new radius
		h = new height
		Sphere Cap Surface = π(a²+ h²)
		Sphere Cap Volume = (1/3)πh²(3R - h)
		Sphere Cap Volume = (1/3)πR³(2 - 3sinα + sin³α)
		https://mathworld.wolfram.com/SphericalCap.html
		"""
		h = -(np.sin(alpha) * self._radius - self._radius)
		a = np.sqrt(h * (2 * self._radius - h))
		return np.pi * (a**2 + h**2)

	def angle(self, p):
		"""
		angle = sin⁻¹((R - h) / R)
		"""
		# point = np.array([*p]) / norm
		h = np.linalg.norm(self._radius - p[-1])
		alpha = np.arcsin(((self._radius - h) / self._radius))
		return alpha

	def random(self, size = 1, alpha = 0):
		"""
		https://mathworld.wolfram.com/SpherePointPicking.html
		https://mathworld.wolfram.com/HyperspherePointPicking.html
		"""
		for i in range(size):
			gauss = np.random.normal(0, 1, self._dim)
			gauss /= np.linalg.norm(gauss)
			point = Point(*(gauss * self._radius))
			angle = self.angle(point.euclidean)
			if ((alpha > 0 and angle > alpha) 
				or (alpha < 0 and angle < alpha)
				or alpha == 0):
				yield point

	def _points(self, size = 100):
		"""
		Get size number of points to graph a n-dim sphere
	
		use ax.plot_wireframe(*HyperSphereObject(), label = "HyperSphere")
		"""
		if self._dim == 1:
			return np.array([self._origin - self._radius, self._origin + self._radius])

		theta, phi = np.mgrid[0:2*np.pi:complex(0, size), 0:np.pi:complex(0, size)]

		if self._dim < 3:
			theta = theta[:,0]
			phi = phi[0,:]

		# sins = np.array([np.sin(phi)**i if i < self._dim-1
		# 								else np.sin(phi)**(i-1*(i != 0))
		# 								for i in reversed(range(self._dim))])

		# coords = np.array([self._origin.euclidean[i] + self._radius * sins[i] 
		# 					* (np.cos(theta) * (i == 0) 
		# 					+ np.sin(theta) * (i == 1) 
		# 					+ np.cos(phi) * (i > 1)) 
		# 					for i in range(self._dim)])

		sins = (np.sin(phi)**i if i < self._dim-1
								else np.sin(phi)**(i-1*(i != 0))
								for i in reversed(range(self._dim)))

		coords = np.array([self._origin.euclidean[i] + self._radius * next(sins) 
							* (np.cos(theta) * (i == 0) 
							+ np.sin(theta) * (i == 1) 
							+ np.cos(phi) * (i > 1)) 
							for i in range(self._dim)])

		if self._dim < 3:
			coords = np.concatenate((coords, [np.zeros(size)]), axis = 0)
			coords = (np.array([coord]) for coord in coords)

		return coords

	def line_intersection(self, v):
		"""
		Intersection of a line with a sphere

		plugging in vector into equation of a sphere
		solving for the intersection point

		https://en.wikipedia.org/wiki/Line–sphere_intersection
		http://paulbourke.net/geometry/circlesphere/
		"""

		if not isinstance(v, Vector):
			if isinstance(v, Point):
				v = Vector(v)
			elif (isinstance(v, np.ndarray) or isinstance(v, list)):
				v = Vector(v)
			else:
				raise ValueError("Norm must be Vector Ojbect or list")

		p1, p2 = v.points
		u = v.direction
		o = v.origin.euclidean
		c = self._origin.euclidean
		r = self._radius

		a = sum(u**2)
		b = 2 * sum((u * (p1 - c)))
		c = sum(c**2) + sum(p1**2) - 2 * sum(c * p1) - r**2

		bb4ac = b**2 - 4 * a * c

		eps = np.finfo(float).eps

		if bb4ac < 0 or abs(a) < eps:
			return (Intersection.line_no_intersect, None)

		a2 = 2 * a
		
		mu1 = (-b - np.sqrt(bb4ac)) / a2
		mu2 = (-b + np.sqrt(bb4ac)) / a2

		# checking different corner cases

		if ((mu1 < 0 and mu2 < 0) or (mu1 > 1 and mu2 > 1)):
			return (Intersection.segment_outside, None)
		
		elif ((mu1 < 0 and mu2 > 1) or (mu2 < 0 and mu1 > 1)):
			return (Intersection.segment_inside, None)
		
		elif ((mu1 >= 0 and mu1 <= 1) and (mu2 < 0 or mu2 > 1)) or ((mu2 >= 0 and mu2 <= 1) and (mu1 < 0 or mu1 > 1)):
			mu = mu2 * (mu2 >= 0 and mu2 <= 1) + mu1 * (not (mu2 >= 0 and mu2 <= 1))
			return (Intersection.segment_partial, Point(*(o + mu * u)))

		elif (abs(mu1 - mu2) < eps and (mu1 >= 0 and mu1 <= 1)):
			return (Intersection.segment_tangent, Point(*(o + mu1 * u)))

		elif ((mu1 >= 0 and mu1 <= 1) and (mu2 >= 0 and mu2 <= 1)):
			return (Intersection.segment_full, (Point(*(o + mu1 * u)), Point(*(o + mu2 * u))))
		else:
			return (Intersection.line_no_intersect, None)

	def reflection(self, line):
		"""
		Reflection of a line segment on the internal sphere surface

		Limitations:
		if the line is close to the tangent of the sphere, and of greate magnitude, reflection point might land outside the sphere due to its curvature.
		"""

		assert isinstance(line, Vector), "Line must be a Vector Object"

		p1, p2 				= line.points
		inters_type, inters = self.line_intersection(line)
		origin 				= self.origin

		inters_len 			= 0

		if inters is not None:
			inters_len 		= len(inters)
		else:
			return inters_type

		if inters_type == Intersection.line_no_intersect:
			return inters_type

		elif inters_type == Intersection.segment_partial:
			pass

		elif inters_type == Intersection.segment_full:
			if isBetween(p1, inters[0], inters[1]) :
				inters = inters[0]
			else:
				inters = inters[1]
			pass

		elif inters_type == Intersection.segment_tangent:
			return inters_type

		elif inters_type == Intersection.segment_inside:
			return inters_type

		elif inters_type == Intersection.segment_outside:
			return inters_type

		if inters.x != 0.0:
			# x axis
			axis = [1.,0.,0.]
			T3 = reflection_YZ_Matrix()
		elif inters.y != 0.0:
			# y axis
			axis = [0.,1.,0.]
			T3 = reflection_XZ_Matrix()
		elif inters.z != 0.0:
			# z axis
			axis = [0.,0.,1.]
			T3 = reflection_XY_Matrix()
		else:
			raise ValueError(f"Intersection is at {Inf}")

		# rotation of the norm of the plane to the xyz-axis
		T1 = rotation(inters, axis)

		# translation matrix to the origin
		T2 = translation(inters, origin)
		
		# transformation matrix compounding
		T = T1 @ T2

		# transformation and reflection of outside point
		dot = Point(*np.dot(T3, np.dot(T, p2.homogeneous)))
		# inverse transformation of outside point
		dot = Point(*np.dot(np.linalg.inv(T), dot.homogeneous))

		# bad bug fix reflection outside point
		if np.linalg.norm(dot.euclidean) > self.radius and np.linalg.norm(p1.euclidean) < self.radius:
			print("recursion")
			return self.reflection(Vector(dot, inters))
			# return Intersection.segment_outside
		
		# return line from intersection to the reflection
		return Vector(dot, inters)

	
