#! /bin/python3

"""
HyperSpace.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.
"""

import numpy as np
import sys

class Point(object):
	"""Homogeneous Point Ojbect Class"""
	
	attrs = ["x","y","z","w"]

	def __init__(self, *coords):
		coords_len 		= len(coords)
		self._coords 	= np.zeros(4)

		if coords_len > 4:
			print("Coords in 3D or Homogeneous 4D available")
		else:
			for i in range(coords_len):
				self._coords[i] = coords[i]
			if coords_len < 4:
				self._coords[-1] = 1
			else:
				self.normalize()

	def __call__(self):
		return self._coords

	def __iter__(self):
		for key in self.euclidean:
			yield key

	def __setitem__(self, key, val):
		self._coords[Point.attrs.index(key)] = val

	def __getitem__(self, key):
		return self._coords[Point.attrs.index(key)]

	def __getattr__(self, attr):
		if attr in Point.attrs:
			return self._coords[Point.attrs.index(attr)]
		elif attr == "euclidean":
			return self._coords[:-1]
		elif attr == "homogeneous":
			return self._coords
		else:
			raise AttributeError(f"{self.__class__.__name__} object has no attribute {attr}")

	def __repr__(self):
		return f"{self._coords}"

	def __str__(self):
		return f"{self._coords[:-1]}"

	def __cmp__(self, other):
		return cmp(self._coords, other.homogeneous)

	def __len__(self):
		return len(self._coords)

	def __add__(self, other):
		if isinstance(other, Point):
			return self.euclidean + other.euclidean
		return self.euclidean + other

	def __sub__(self, other):
		if isinstance(other, Point):
			return self.euclidean - other.euclidean
		return self.euclidean - other
		
	def __mul__(self, other):
		if isinstance(other, Point):
			return self.euclidean * other.euclidean
		return self.euclidean * other
		
	def __truediv__(self, other):
		if isinstance(other, Point):
			return self.euclidean / other.euclidean
		return self.euclidean / other
		
	def __floordiv__(self, other):
		if isinstance(other, Point):
			return self.euclidean // other.euclidean
		return self.euclidean // other
		
	def __mod__(self, other):
		if isinstance(other, Point):
			return self.euclidean % other.euclidean
		return self.euclidean % other
		
	def __pow__(self, other):
		if isinstance(other, Point):
			return self.euclidean ** other.euclidean
		return self.euclidean ** other
		
	def __rshift__(self, other):
		if isinstance(other, Point):
			return self.euclidean >> other.euclidean
		return self.euclidean >> other
		
	def __lshift__(self, other):
		if isinstance(other, Point):
			return self.euclidean << other.euclidean
		return self.euclidean << other
		
	def __and__(self, other):
		if isinstance(other, Point):
			return self.euclidean & other.euclidean
		return self.euclidean & other
		
	def __or__(self, other):
		if isinstance(other, Point):
			return self.euclidean | other.euclidean
		return self.euclidean | other
		
	def __xor__(self, other):
		if isinstance(other, Point):
			return self.euclidean ^ other.euclidean
		return self.euclidean ^ other

	def __iadd__(self, other):
		if isinstance(other, Point):
			self.euclidean += other.euclidean
		else:
			self.euclidean += other
		return self

	def __isub__(self, other):
		if isinstance(other, Point):
			self.euclidean -= other.euclidean
		else:
			self.euclidean -= other
		return self
		
	def __imul__(self, other):
		if isinstance(other, Point):
			self.euclidean *= other.euclidean
		else:
			self.euclidean *= other
		return self
		
	def __itruediv__(self, other):
		if isinstance(other, Point):
			self.euclidean /= other.euclidean
		else:
			self.euclidean /= other
		return self
		
	def __ifloordiv__(self, other):
		if isinstance(other, Point):
			self.euclidean //= other.euclidean
		else:
			self.euclidean //= other
		return self
		
	def __imod__(self, other):
		if isinstance(other, Point):
			self.euclidean %= other.euclidean
		else:
			self.euclidean %= other
		return self
		
	def __ipow__(self, other):
		if isinstance(other, Point):
			self.euclidean **= other.euclidean
		else:
			self.euclidean **= other
		return self
		
	def __irshift__(self, other):
		if isinstance(other, Point):
			self.euclidean >>= other.euclidean
		else:
			self.euclidean >>= other
		return self
		
	def __ilshift__(self, other):
		if isinstance(other, Point):
			self.euclidean <<= other.euclidean
		else:
			self.euclidean <<= other
		return self
		
	def __iand__(self, other):
		if isinstance(other, Point):
			self.euclidean &= other.euclidean
		else:
			self.euclidean &= other
		return self
		
	def __ior__(self, other):
		if isinstance(other, Point):
			self.euclidean |= other.euclidean
		else:
			self.euclidean |= other
		return self
		
	def __ixor__(self, other):
		if isinstance(other, Point):
			self.euclidean ^= other.euclidean
		else:
			self.euclidean ^= other
		return self

	def __lt__(self, other):
		return np.all(self._coords < other.homogeneous)

	def __gt__(self, other):
		return np.all(self._coords > other.homogeneous)

	def __le__(self, other):
		return np.all(self._coords <= other.homogeneous)

	def __ge__(self, other):
		return np.all(self._coords >= other.homogeneous)

	def __eq__(self, other):
		return np.all(self._coords == other.homogeneous)

	def __ne__(self, other):
		return np.all(self._coords != other.homogeneous)
		
	def normalize(self):
		if self._coords[-1] != 0:
			self._coords /= self._coords[-1]


class Vector(object):
	"""Vector Ojbect Class"""

	def __init__(self, p1, p0 = None):
		if not isinstance(p1, Point):
			if (isinstance(p1, np.ndarray) or isinstance(p1, list)):
				p1 = Point(*p1)
			else:
				raise ValueError("Point 1 must be Point Ojbect or list")

		if not isinstance(p0, Point):
			if (isinstance(p0, np.ndarray) or isinstance(p0, list)):
				p0 = Point(*p0)
			elif p0 == None:
				p0 = Point()
			else:
				raise ValueError("Point 0 must be Point Ojbect or list or None")

		self._points 	= [p0, p1]
		self._dim 		= len(p1) - 1

	def __call__(self):
		return self.moment

	def __iter__(self):
		for point in self.euclidean:
			yield point

	def __getattr__(self, attr):
		if attr == "points":
			return self._points
		elif attr == "euclidean":
			return np.array([self._points[0].euclidean, self._points[1].euclidean])
		elif attr == "homogeneous":
			return np.array([self._points[0].homogeneous, self._points[1].homogeneous])
		elif attr == "origin":
			return self._points[0]
		elif attr == "moment":
			diff = self._points[1] - self._points[0]
			diff /= np.linalg.norm(diff)
			return Point(*(diff))
		elif attr == "direction":
			return Point(*(self._points[1] - self._points[0]))
		elif attr == "len":
			return np.linalg.norm(self._points[1].homogeneous - self._points[0].homogeneous)
		elif attr == "coefs":
			return Point(*np.cross(*self._points))
		else:
			raise AttributeError(f"{self.__class__.__name__} object has no attribute {attr}")

	def __repr__(self):
		return f"{self.homogeneous}"

	def __str__(self):
		return f"{np.array([*self._points])}"

	def __cmp__(self, other):
		return cmp(self.homogeneous, other.homogeneous)

	def __eq__(self, other):
		return np.all(self.homogeneous == other.homogeneous)

	def __ne__(self, other):
		return np.all(self.homogeneous != other.homogeneous)

	def __len__(self):
		return len(self._points)

	def __add__(self, other):
		if isinstance(other, Vector):
			return self.moment + other.moment
		return self.moment + other

	def __sub__(self, other):
		if isinstance(other, Vector):
			return self.moment - other.moment
		return self.moment - other
		
	def __mul__(self, other):
		if isinstance(other, Vector):
			return self.moment * other.moment
		return self.moment * other
		
	def __truediv__(self, other):
		if isinstance(other, Vector):
			return self.moment / other.moment
		return self.moment / other
		
	def __floordiv__(self, other):
		if isinstance(other, Vector):
			return self.moment // other.moment
		return self.moment // other
		
	def __mod__(self, other):
		if isinstance(other, Vector):
			return self.moment % other.moment
		return self.moment % other
		
	def __pow__(self, other):
		if isinstance(other, Vector):
			return self.moment ** other.moment
		return self.moment ** other
		
	def __rshift__(self, other):
		if isinstance(other, Vector):
			return self.moment >> other.moment
		return self.moment >> other
		
	def __lshift__(self, other):
		if isinstance(other, Vector):
			return self.moment << other.moment
		return self.moment << other
		
	def __and__(self, other):
		if isinstance(other, Vector):
			return self.moment & other.moment
		return self.moment & other
		
	def __or__(self, other):
		if isinstance(other, Vector):
			return self.moment | other.moment
		return self.moment | other
		
	def __xor__(self, other):
		if isinstance(other, Vector):
			return self.moment ^ other.moment
		return self.moment ^ other

	def __iadd__(self, other):
		if isinstance(other, Vector):
			self._points[0] += other.moment
			self._points[1] += other.moment
		else:
			self._points[0] += other
			self._points[1] += other
		return self

	def __isub__(self, other):
		if isinstance(other, Vector):
			self._points[0] -= other.moment
			self._points[1] -= other.moment
		else:
			self._points[0] -= other
			self._points[1] -= other
		return self
		
	def __imul__(self, other):
		if isinstance(other, Vector):
			self._points[0] *= other.moment
			self._points[1] *= other.moment
		else:
			self._points[0] *= other
			self._points[1] *= other
		return self
		
	def __itruediv__(self, other):
		if isinstance(other, Vector):
			self._points[0] /= other.moment
			self._points[1] /= other.moment
		else:
			self._points[0] /= other
			self._points[1] /= other
		return self
		
	def __ifloordiv__(self, other):
		if isinstance(other, Vector):
			self._points[0] //= other.moment
			self._points[1] //= other.moment
		else:
			self._points[0] //= other
			self._points[1] //= other
		return self
		
	def __imod__(self, other):
		if isinstance(other, Vector):
			self._points[0] %= other.moment
			self._points[1] %= other.moment
		else:
			self._points[0] %= other
			self._points[1] %= other
		return self
		
	def __ipow__(self, other):
		if isinstance(other, Vector):
			self._points[0] **= other.moment
			self._points[1] **= other.moment
		else:
			self._points[0] **= other
			self._points[1] **= other
		return self
		
	def __irshift__(self, other):
		if isinstance(other, Vector):
			self._points[0] >>= other.moment
			self._points[1] >>= other.moment
		else:
			self._points[0] >>= other
			self._points[1] >>= other
		return self
		
	def __ilshift__(self, other):
		if isinstance(other, Vector):
			self._points[0] <<= other.moment
			self._points[1] <<= other.moment
		else:
			self._points[0] <<= other
			self._points[1] <<= other
		return self
		
	def __iand__(self, other):
		if isinstance(other, Vector):
			self._points[0] &= other.moment
			self._points[1] &= other.moment
		else:
			self._points[0] &= other
			self._points[1] &= other
		return self
		
	def __ior__(self, other):
		if isinstance(other, Vector):
			self._points[0] |= other.moment
			self._points[1] |= other.moment
		else:
			self._points[0] |= other
			self._points[1] |= other
		return self
		
	def __ixor__(self, other):
		if isinstance(other, Vector):
			self._points[0] ^= other.moment
			self._points[1] ^= other.moment
		else:
			self._points[0] ^= other
			self._points[1] ^= other
		return self

	def __lt__(self, other):
		return np.all(self.moment < other.moment)

	def __gt__(self, other):
		return np.all(self.moment > other._moment)

	def __le__(self, other):
		return np.all(self.moment <= other.moment)

	def __ge__(self, other):
		return np.all(self.moment >= other.moment)

	def __eq__(self, other):
		if isinstance(other, Vector):
			return np.all(self.moment == other.moment)
		return False

	def __ne__(self, other):
		if isinstance(other, Vector):
			return np.all(self.moment != other.moment)
		return True

	def distance(self, p, on = False):
		if not isinstance(p, Point):
			if (isinstance(p, np.ndarray) or isinstance(p, list)):
				p = Point(*p)
			else:
				raise ValueError("Point must be Point Ojbect or list")

		# perpendicular distance from point to vector
		d_perp = np.dot(self.coefs, p.euclidean)

		# if point on vector and on == True return distance from origin (p0)
		if on == True and abs(d_perp) < np.finfo(float).eps * 1e4:
			return np.linalg.norm(p - self.origin)

		return d_perp

	def intersection(self, v):
		if not isinstance(v, Vector):
			if (isinstance(v, np.ndarray) or isinstance(v, list)):
				v = Vector(*v)
			else:
				raise ValueError("Vector must be Vector Ojbect or list")			
		return np.cross(self.coefs, v.coefs)

	def slope(self):
		return (self._points[1].y - self._points[0].y) / (self._points[1].x - self._points[0].x)

	def intercept(self):
		return self._points[0].y - self.slope() * self._points[0].x

	def point(self, d):
		return self.origin + self.moment * d

	def skew(self):
		"""
		Skew Symmetric Matrix
		"""
		x = self.coefs.x
		y = self.coefs.y
		z = self.coefs.z
		return np.array([[ 0, -z,  y],
						 [ z,  0, -x],
						 [-y,  x,  0]])


class Plane(object):
	"""Plane Object Class"""

	def __init__(self, norm, origin = None):
		if not isinstance(norm, Vector):
			if isinstance(norm, Point):
				self._norm = Vector(norm, origin)
			elif (isinstance(norm, np.ndarray) or isinstance(norm, list)):
				self._norm = Vector(norm, origin)
			else:
				raise ValueError("Norm must be Vector Ojbect or list")
		else:
			self._norm = norm

		if not isinstance(origin, Point):
			if (isinstance(origin, np.ndarray) or isinstance(origin, list)):
				self._origin = Point(*origin)
			elif origin == None:
				self._origin = Point()
			else:
				raise ValueError("Origin must be Point Ojbect or list or None")
		else:
			self._origin = origin


		





