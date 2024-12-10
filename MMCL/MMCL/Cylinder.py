#! /bin/python3

"""
Cylinder.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.
"""
from MMCL.Transformations 	import *

from MMCL.Utils 			import base10_exponent, isBetween
from MMCL.HyperSpace 		import Point, Vector, Plane

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
import sys

class Cylinder(object):
	"""
	Cylinder Object Class
	"""

	def __init__(self, radius=1., height=1., origin=None):
		self._radius 	= float(radius)
		self._height 	= float(height)
		
		if not isinstance(origin, Point):
			if (isinstance(origin, np.ndarray) or isinstance(origin, list)):
				self._origin = Point(*origin)
			elif origin == None:
				self._origin = Point()
			else:
				raise ValueError("Origin must be Point Ojbect or list or None")
		else:
			self._origin = origin

	def __call__(self, orientation='z', circle_dx=100, height_dx=100):
		return self._points(orientation, circle_dx, height_dx)

	def __iter__(self):
		for point in self._points():
			yield point

	def __getattr__(self, attr):
		if attr == "radius":
			return self._radius
		elif attr == "height":
			return self._height
		elif attr == "origin":
			return self._origin
		else:
			raise AttributeError(f"{self.__class__.__name__} object has no attribute {attr}")

	def __repr__(self):
		return f"R = {self._radius}, dim = {self._dim}, origin = {self._origin.homogeneous}"

	def __str__(self):
		return f"R = {self._radius}, dim = {self._dim}, origin = {self._origin.euclidean}"

	def _points(self, orientation='z', circle_dx=100, height_dx=100):
		"""
		Get size number of points to graph a cylinder
	
		use ax.plot_wireframe(*CylinderObject(), label = "Cylinder")
		"""
		z = np.linspace(0, self._height, height_dx)
		theta = np.linspace(0, 2*np.pi, circle_dx)
		theta_grid, z_grid = np.meshgrid(theta, z)
		if orientation == 'z':
			coords = np.array([
				self._radius * np.cos(theta_grid) + self._origin.x,
				self._radius * np.sin(theta_grid) + self._origin.y,
				z_grid + self._origin.z
			])
		elif orientation == 'y':
			coords = np.array([
				self._radius * np.cos(theta_grid) + self._origin.x,
				z_grid + self._origin.y,
				self._radius * np.sin(theta_grid) + self._origin.z,
			])
		elif orientation == 'x':
			coords = np.array([
				z_grid + self._origin.x,
				self._radius * np.cos(theta_grid) + self._origin.y,
				self._radius * np.sin(theta_grid) + self._origin.z,
			])
		else:
			raise ValueError

		return coords

	def scatter(self, density, max_d, orientation='z', height_dx=100):
		norm = density * self._height / height_dx
		norm = np.nan_to_num(norm, nan=0)


		scale = base10_exponent(max_d)
		norm *= 10**(2-scale)
		norm = norm.astype(int)

		points = []
		colors = []
		for i in range(len(density)):
			num_points = norm[i]
			theta = np.random.uniform(0, 2 * np.pi, num_points)

			r = self._radius * np.sqrt(np.random.uniform(0, 1, num_points))
			if orientation == 'z':
				x = r * np.cos(theta) + self._origin.x
				y = r * np.sin(theta) + self._origin.y
				z = np.full(num_points, i * (self._height / len(density))) + self._origin.z
			elif orientation == 'y':
				x = r * np.cos(theta) + self._origin.x
				z = r * np.sin(theta) + self._origin.z
				y = np.full(num_points, i * (self._height / len(density))) + self._origin.y
			elif orientation == 'x':
				z = r * np.cos(theta) + self._origin.z
				y = r * np.sin(theta) + self._origin.y
				x = np.full(num_points, i * (self._height / len(density))) + self._origin.x
			else:
				raise ValueError

			points.append(np.vstack((x, y, z)))
			colors.append(np.full_like(x, density[i]))

		points = np.hstack(points).T 
		colors = np.hstack(colors).T
		return (points[:, 0], points[:, 1], points[:, 2]), colors

	
