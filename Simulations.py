#! /bin/python3

"""
Simulations.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.
"""

from MMCL import *

# from Transformations 		import *
# from MFPT 					import *

# from Utils 					import percentageDifference, writeCSV, readCSV
# from Brownian 				import brownian
# from HyperSpace 			import Point, Vector, Plane
# from HyperSphere 			import Intersection, HyperSphere
# from Polygon 				import Polygon
# from CamKII 				import CamKII

from matplotlib 			import pyplot as plt
# import matplotlib.animation as animation
from mpl_toolkits.mplot3d 	import Axes3D

from multiprocessing 		import Process, Queue

import holoviews as hv
import numpy as np
import warnings
import time
import sys

hv.extension('matplotlib')
warnings.filterwarnings('ignore') #temporary fix to suppress deprecation warnings we get with matplotlib


def simplified_simulation_sphere_in_sphere_thread(queue, queue2, D, N_sim, R1, R2, dim = 3, dt = 1, pid = 0, tolerance = 1e4):
	sphere = HyperSphere(R2, dim)
	subsphere = CamKII.Subunit(R1, dim)

	times = []

	mfpt = 0

	for i in range(int(N_sim)):

		brow = [next(sphere.random(size = 1))]

		brow.append(Vector(brow[0], sphere.origin).point(R2 * 0.9))

		inter_point = Point(*[0,0,0])

		index = 1

		inter_type, inters = sphere.line_intersection(Vector(brow[1], brow[0]))
	
		while True:
			b = next(brownian(D = D, dt = dt, d = dim, origin = brow[-1]))
			ref_small = subsphere.reflection(b)
			if isinstance(ref_small, Vector):
				brow.append(ref_small.points[0])
				subsphere.intersection_point = ref_small.points[0]
				subsphere.activated = True
				break
			ref_big = sphere.reflection(b)
			if isinstance(ref_big, Vector):
				b = ref_big
				if sphere.radius < np.linalg.norm(ref_big.euclidean[1]):
					print(f"outside")
					queue2.put(1)
					break
				brow.append(b.points[1])
			else:
				if ref_big == Intersection.segment_inside or ref_big == Intersection.line_no_intersect:
					brow.append(b.points[1])

			if index > tolerance:
				print("Fail")
				queue2.put(1)
				break
			index += 1
		
		if index <= tolerance or sphere.radius >= np.linalg.norm(b.euclidean[1]):
			times.append(len(brow))
		else:
			print(f"{pid}: iteration: {i} -> {len(brow)}")

	queue.put([np.mean(times)])


def simulation_sphere_in_sphere_threadpool(D, R1, R2, dt, n_threads, N_sim, dim):
	threads = []

	queue = Queue()
	queue2 = Queue()

	n_sim = N_sim / n_threads

	mfpts = []

	start = time.time()

	for i in range(n_threads):
		threads.append(Process(target=simplified_simulation_sphere_in_sphere_thread, args=(queue, queue2, D, n_sim, R1, R2, dim, dt, i, 1e9)))
		threads[i].start()

	for i in range(n_threads):
		threads[i].join()

	for i in range(n_threads):
		mfpts = mfpts + queue.get()

	failed = 0
	while not queue2.empty():
		failed += queue2.get()

	end = time.time()
	print("time:")
	print(end - start)


	if dim == 2:
		mfpt_analytic = mfpt_analytic_2D(D, R1, R2)
	if dim == 3:
		mfpt_analytic = mfpt_analytic_3D(D, R1, R2)

	sim_time = np.mean(mfpts) * dt
		

	return {
		"D": D,
		"R1": R1,
		"R2": R2,
		"dt": dt,
		"N_sim": N_sim,
		"d": dim,
		"exec_time": end - start,
		"sim_time": sim_time,
		"anali_time": mfpt_analytic,
		"percent_diff": percentageDifference(mfpt_analytic, sim_time),
		"failed": failed
	}

def simplified_simulation_sphere_in_sphere_plot(D, R1, R2, dim = 3, dt = 1, tolerance = 1e4):

	sphere = HyperSphere(R2, dim)
	subsphere = CamKII.Subunit(R1, dim)

	brow = [next(sphere.random(size = 1))]

	brow.append(Vector(brow[0], sphere.origin).point(R2 * 0.9))

	inter_point = Point(*[0,0,0])

	index = 1

	while True:
		if index % 1000 == 0:
			print("index", index)
		dt = adjustable_dt(Vector(brow[-1], brow[-2]).distance(Vector(brow[-1], subsphere.origin).point(R1), on = True))
		b = next(brownian(D = D, dt = dt, d = dim, origin = brow[-1]))
		ref_small = subsphere.reflection(b)
		if isinstance(ref_small, Vector):
			brow.append(ref_small.points[0])
			subsphere.intersection_point = ref_small.points[0]
			subsphere.activated = True
			break
		ref_big = sphere.reflection(b)
		if isinstance(ref_big, Vector):
			brow.append(ref_big.points[1])
		else:
			if ref_big == Intersection.segment_inside or ref_big == Intersection.line_no_intersect:
				brow.append(b.points[1])

		if index > tolerance:
			print("Fail")
			break
		index += 1

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.set_title("Brownian 3D")
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	ax.set_autoscale_on(True)

	ax.plot_wireframe(*sphere(10 + (90 * (dim == 2))), label = "Sphere", color='b')

	if subsphere.activated:
		ax.plot_wireframe(*subsphere(10 + (90 * (dim == 2))), label = "SubSphere", color = 'y')
	else:
		ax.plot_wireframe(*subsphere(10 + (90 * (dim == 2))), label = "SubSphere", color = 'g')
	ax.plot(*np.array(brow).T, color = 'c')

	# brownian starting point
	ax.scatter(*brow[0].euclidean.T, color = 'r')
	ax.scatter(*brow[1].T, color = 'r')
	if isinstance(subsphere.intersection_point, Point):	
		# brownian intersection point
		ax.scatter(*subsphere.intersection_point.euclidean.T, color = 'r')

	ax.legend()
	plt.show()

	# forloop used with %matplotlib notebook for interaction with the figure
	for angle in range(0, 360):
		ax.view_init(30, angle)
		plt.draw()
		plt.pause(.001)

def simplified_simulation_CAMKII_1_sphere_in_sphere_plot(D, R2, dim = 3, dt = 1, tolerance = 1e4):
	sphere = HyperSphere(R2, dim)

	brow = [next(sphere.random(size = 1))]

	brow.append(Vector(brow[0], sphere.origin).point(R2 * 0.9))

	inter_point = Point(*[0,0,0])

	index = 1

	camkii = CamKII(6, 3, 0, dim = dim)

	while True:
		b = next(brownian(D = D, dt = dt, d = dim, origin = brow[-1]))
		dist = []
		for subunit in camkii:
			if isinstance(subunit, CamKII.Subunit):
				ref_small = subunit.reflection(b)
				if isinstance(ref_small, Vector):
					dist.append((b.distance(ref_small.points[0], on = True), subunit, ref_small))

		if len(dist) > 0:
			# calculate the minimum distance of intersection to sphere
			min_d = min([d for d,i,j in dist])
			# store only the closest sphere
			s = [i for d,i,j in dist if d == min_d][0]
			inters = [j for d,i,j in dist if d == min_d][0]
			brow.append(inters.points[0])
			s.intersection_point = inters.points[0]
			s.activated = True
			break
	
		ref_big = sphere.reflection(b)
		if isinstance(ref_big, Vector):
			if sphere.radius < np.linalg.norm(ref_big.euclidean[1]):
				print(f"outside")
				break
			brow.append(ref_big.points[1])
		else:
			if ref_big == Intersection.segment_inside or ref_big == Intersection.line_no_intersect:
				brow.append(b.points[1])

		if index > tolerance:
			print("Fail")
			break
		index += 1

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.set_title("Brownian 3D")
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	ax.set_autoscale_on(True)

	ax.plot_wireframe(*sphere(10 + (90 * (dim == 2))), label = "Sphere", color='b')

	for subunit in camkii:
		if isinstance(subunit, CamKII.Subunit):
			if subunit.activated:
				ax.plot_wireframe(*subunit(10 + (90 * (dim == 2))), color = 'y')
				if isinstance(subunit.intersection_point, Point):	
					# brownian intersection point
					ax.scatter(*subunit.intersection_point.euclidean.T, color = 'r')
			else:
				ax.plot_wireframe(*subunit(10 + (90 * (dim == 2))), color = 'g')
		elif isinstance(subunit, Vector):
			ax.plot(*subunit.euclidean.T, color = 'g')

	ax.plot(*np.array(brow).T, color = 'c')

	# brownian starting point
	ax.scatter(*brow[0].euclidean.T, color = 'r')
	ax.scatter(*brow[1].T, color = 'r')

	ax.legend()
	plt.show()

	# forloop used with %matplotlib notebook for interaction with the figure
	for angle in range(0, 360):
		ax.view_init(30, angle)
		plt.draw()
		plt.pause(.001)

def simplified_simulation_CAMKII_2_sphere_in_sphere_plot(D, R2, dim = 3, dt = 1, tolerance = 1e4):
	sphere = HyperSphere(R2, dim)

	brow = [next(sphere.random(size = 1))]

	brow.append(Vector(brow[0], sphere.origin).point(R2 * 0.9))

	inter_point = Point(*[0,0,0])

	index = 1

	camkii = CamKII(6, 3, 0, dim = dim)

	for subunit in camkii:
		if isinstance(subunit, CamKII.Subunit):
			subunit.activated = True
			break

	while True:
		b = next(brownian(D = D, dt = dt, d = dim, origin = brow[-1]))
		dist = []
		for subunit in camkii:
			if isinstance(subunit, CamKII.Subunit):
				ref_small = subunit.reflection(b)
				if isinstance(ref_small, Vector):
					dist.append((b.distance(ref_small.points[0], on = True), subunit, ref_small))

		if len(dist) > 0:
			# calculate the minimum distance of intersection to sphere
			min_d = min([d for d,i,j in dist])
			# store only the closest sphere
			s = [i for d,i,j in dist if d == min_d][0]
			inters = [j for d,i,j in dist if d == min_d][0]

			if s.activated:
				brow.append(inters.points[1])
			else:
				brow.append(inters.points[0])
				s.intersection_point = inters.points[0]
				s.activated = True
				break
	
		ref_big = sphere.reflection(b)
		if isinstance(ref_big, Vector):
			if sphere.radius < np.linalg.norm(ref_big.euclidean[1]):
				print(f"outside")
				break
			brow.append(ref_big.points[1])
		else:
			if ref_big == Intersection.segment_inside or ref_big == Intersection.line_no_intersect:
				brow.append(b.points[1])

		if index > tolerance:
			print("Fail")
			break
		index += 1

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.set_title("Brownian 3D")
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	ax.set_autoscale_on(True)

	ax.plot_wireframe(*sphere(10 + (90 * (dim == 2))), label = "Sphere", color='b')

	for subunit in camkii:
		if isinstance(subunit, CamKII.Subunit):
			if subunit.activated:
				print(subunit.activated)
				print(subunit)
				ax.plot_wireframe(*subunit(10 + (90 * (dim == 2))), color = 'y')
				if isinstance(subunit.intersection_point, Point):	
					# brownian intersection point
					ax.scatter(*subunit.intersection_point.euclidean.T, color = 'r')
			else:
				print(subunit.activated)
				print(subunit)
				ax.plot_wireframe(*subunit(10 + (90 * (dim == 2))), color = 'g')
		elif isinstance(subunit, Vector):
			ax.plot(*subunit.euclidean.T, color = 'g')

	ax.plot(*np.array(brow).T, color = 'c')

	# brownian starting point
	ax.scatter(*brow[0].euclidean.T, color = 'r')
	ax.scatter(*brow[1].T, color = 'r')

	ax.legend()
	plt.show()

	# forloop used with %matplotlib notebook for interaction with the figure
	for angle in range(0, 360):
		ax.view_init(30, angle)
		plt.draw()
		plt.pause(.001)

def simplified_simulation_CAMKII_3_sphere_in_sphere_plot(D, R2, dim = 3, dt = 1, tolerance = 1e4):
	sphere = HyperSphere(R2, dim)

	brow = [next(sphere.random(size = 1))]

	brow.append(Vector(brow[0], sphere.origin).point(R2 * 0.9))

	inter_point = Point(*[0,0,0])

	index = 1

	camkii = CamKII(6, 3, 0, dim = dim)

	subunits = [i for i in camkii]

	counter = 0
	subunits[0].activated = True
	subunits[1].activated = True
	subunits[5].activated = True

	while True:
		b = next(brownian(D = D, dt = dt, d = dim, origin = brow[-1]))
		dist = []
		for subunit in camkii:
			if isinstance(subunit, CamKII.Subunit):
				ref_small = subunit.reflection(b)
				if isinstance(ref_small, Vector):
					dist.append((b.distance(ref_small.points[0], on = True), subunit, ref_small))

		if len(dist) > 0:
			# calculate the minimum distance of intersection to sphere
			min_d = min([d for d,i,j in dist])
			# store only the closest sphere
			s = [i for d,i,j in dist if d == min_d][0]
			inters = [j for d,i,j in dist if d == min_d][0]

			if s.activated:
				brow.append(inters.points[1])
			else:
				brow.append(inters.points[0])
				s.intersection_point = inters.points[0]
				s.activated = True
				break
	
		ref_big = sphere.reflection(b)
		if isinstance(ref_big, Vector):
			if sphere.radius < np.linalg.norm(ref_big.euclidean[1]):
				print(f"outside")
				break
			brow.append(ref_big.points[1])
		else:
			if ref_big == Intersection.segment_inside or ref_big == Intersection.line_no_intersect:
				brow.append(b.points[1])

		if index > tolerance:
			print("Fail")
			break
		index += 1

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.set_title("Brownian 3D")
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	ax.set_autoscale_on(True)

	ax.plot_wireframe(*sphere(10 + (90 * (dim == 2))), label = "Sphere", color='b')

	for subunit in camkii:
		if isinstance(subunit, CamKII.Subunit):
			if subunit.activated:
				print(subunit.activated)
				print(subunit)
				ax.plot_wireframe(*subunit(10 + (90 * (dim == 2))), color = 'y')
				if isinstance(subunit.intersection_point, Point):	
					# brownian intersection point
					ax.scatter(*subunit.intersection_point.euclidean.T, color = 'r')
			else:
				print(subunit.activated)
				print(subunit)
				ax.plot_wireframe(*subunit(10 + (90 * (dim == 2))), color = 'g')
		elif isinstance(subunit, Vector):
			ax.plot(*subunit.euclidean.T, color = 'g')

	ax.plot(*np.array(brow).T, color = 'c')

	# brownian starting point
	ax.scatter(*brow[0].euclidean.T, color = 'r')
	ax.scatter(*brow[1].T, color = 'r')

	ax.legend()
	plt.show()

	# forloop used with %matplotlib notebook for interaction with the figure
	for angle in range(0, 360):
		ax.view_init(30, angle)
		plt.draw()
		plt.pause(.001)

def simplified_simulation_CAMKII_4_sphere_in_sphere_plot(D, R2, dim = 3, dt = 1, tolerance = 1e4):
	sphere = HyperSphere(R2, dim)

	brow = [next(sphere.random(size = 1))]

	brow.append(Vector(brow[0], sphere.origin).point(R2 * 0.9))

	inter_point = Point(*[0,0,0])

	index = 1

	camkii = CamKII(6, 3, 0, dim = dim)

	subunits = [i for i in camkii]

	counter = 0
	subunits[0].activated = True
	subunits[1].activated = True
	subunits[2].activated = True
	subunits[4].activated = True
	subunits[5].activated = True

	while True:
		b = next(brownian(D = D, dt = dt, d = dim, origin = brow[-1]))
		dist = []
		for subunit in camkii:
			if isinstance(subunit, CamKII.Subunit):
				ref_small = subunit.reflection(b)
				if isinstance(ref_small, Vector):
					dist.append((b.distance(ref_small.points[0], on = True), subunit, ref_small))

		if len(dist) > 0:
			# calculate the minimum distance of intersection to sphere
			min_d = min([d for d,i,j in dist])
			# store only the closest sphere
			s = [i for d,i,j in dist if d == min_d][0]
			inters = [j for d,i,j in dist if d == min_d][0]

			if s.activated:
				brow.append(inters.points[1])
			else:
				brow.append(inters.points[0])
				s.intersection_point = inters.points[0]
				s.activated = True
				break
	
		ref_big = sphere.reflection(b)
		if isinstance(ref_big, Vector):
			if sphere.radius < np.linalg.norm(ref_big.euclidean[1]):
				print(f"outside")
				break
			brow.append(ref_big.points[1])
		else:
			if ref_big == Intersection.segment_inside or ref_big == Intersection.line_no_intersect:
				brow.append(b.points[1])

		if index > tolerance:
			print("Fail")
			break
		index += 1

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.set_title("Brownian 3D")
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	ax.set_autoscale_on(True)

	ax.plot_wireframe(*sphere(10 + (90 * (dim == 2))), label = "Sphere", color='b')

	for subunit in camkii:
		if isinstance(subunit, CamKII.Subunit):
			if subunit.activated:
				print(subunit.activated)
				print(subunit)
				ax.plot_wireframe(*subunit(10 + (90 * (dim == 2))), color = 'y')
				if isinstance(subunit.intersection_point, Point):	
					# brownian intersection point
					ax.scatter(*subunit.intersection_point.euclidean.T, color = 'r')
			else:
				print(subunit.activated)
				print(subunit)
				ax.plot_wireframe(*subunit(10 + (90 * (dim == 2))), color = 'g')
		elif isinstance(subunit, Vector):
			ax.plot(*subunit.euclidean.T, color = 'g')

	ax.plot(*np.array(brow).T, color = 'c')

	# brownian starting point
	ax.scatter(*brow[0].euclidean.T, color = 'r')
	ax.scatter(*brow[1].T, color = 'r')

	ax.legend()
	plt.show()

	# forloop used with %matplotlib notebook for interaction with the figure
	for angle in range(0, 360):
		ax.view_init(30, angle)
		plt.draw()
		plt.pause(.001)

def real_simulation_plot():
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.set_title("Brownian 3D")
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	R = 100
	dim = 3

	p0 = Point(-25,-50,-25)
	p1 = Point(100,100,100)
	line = Vector(p1, p0)

	sphere = HyperSphere(R, dim)
	sphere_sub = HyperSphere(R * 0.9, dim)

	mfpt = MFPT(3, 10, 0.1, sphere)

	ax.set_autoscale_on(True)
	# ax.set_xlim(4, 8)
	# ax.set_ylim(15, 20)
	# ax.set_zlim(-15,-25)


	# # shpere and reflection
	ax.plot_wireframe(*sphere(10), label = "Sphere", color='b')
	# ax.plot_wireframe(*sphere_sub(10), label = "Sphere", color='c')


	# # CamKII object
	# camkii = list(CamKII(6, 3, 6))

	# # Random Points
	points = list(sphere.random(100, np.pi/3))[:3]
	if points != []:
		boo = False
		sphere_inter = []
		camkiis = []
		ax.scatter(*np.array(points).T, color = 'r')
		for point in points:
			# print(point)
			v = Vector(point, sphere.origin)
			p2 = v.point(sphere_sub.radius)
			# print(p2)
			# ax.scatter(*p2.T, color = "g")

			brow = [point]
			brow.append(p2)

			cam = [Point(0,0,0)]

			for i in range(10000):
				cam_brow = brownian2(D = 5, T = 2, dt = 1, d = 3, origin = cam[i])
				cam_b = next(cam_brow)
				bb = brownian2(D = 10, T = 2, dt = 1, d = 3, origin = brow[i+1])
				b = next(bb)
				ref = reflection(sphere, b)
				if isinstance(ref, Vector):
					b = ref
				brow.append(b.points[1])

				ref = reflection(sphere, cam_b)
				if isinstance(ref, Vector):
					cam_b = ref
				cam.append(cam_b.points[1])

				camkii = CamKII(6, 3, 6, origin = cam[i])
				dist = []

				for j in camkii:
					if isinstance(j, HyperSphere):
						tp, inters = j.line_intersection(b)
						# print(i, tp)
						if tp == Intersection.segment_partial or tp == Intersection.segment_full:
							print(tp, inters)
							if tp == Intersection.segment_full:
								dist.append((b.distance(inters[0], on = True), j))
								dist.append((b.distance(inters[1], on = True), j))
							else:
								dist.append((b.distance(inters, on = True), j))
							boo = True
					elif isinstance(j, Vector):
						continue
					else:
						raise ValueError("CamKII should return Objects")
						
				if boo == True:
					min_d = min([d for d,i in dist])
					s = [i for d,i in dist if d == min_d][0]
					sphere_inter.append(s)
					print("boo == True")
					for j in camkii:
						if isinstance(j, HyperSphere):
							tp, inters = j.line_intersection(b)
							if tp == Intersection.segment_partial or tp == Intersection.segment_full:
								ax.plot_wireframe(*j(10), color = 'b')
								if j == s:
									ax.plot_wireframe(*s(10), color = 'y')
									if tp == Intersection.segment_full:
										if b.distance(inters[0], on = True) == min_d:
											b = Vector(inters[0], b.origin)
											ax.scatter(*inters[0].euclidean.T, color = 'r')
										else:
											b = Vector(inters[1], b.origin)
											ax.scatter(*inters[1].euclidean.T, color = 'r')
									else:
										b = Vector(inters, b.origin)
										ax.scatter(*inters.euclidean.T, color = 'r')
							else:
								continue
								# ax.plot_wireframe(*j(10), color = 'b')
						elif isinstance(j, Vector):
							continue
							# ax.plot(*j.euclidean.T, color = 'b')
						else:
							raise ValueError("CamKII should return Objects")
					boo = False
					camkiis.append(camkii)
					break

			ax.plot(*np.array(brow).T)
		
		print("camkiis")
		print(camkiis)
		for camkii in camkiis:
			print(sphere_inter)
			for j in camkii:
				if isinstance(j, HyperSphere):
					if j not in sphere_inter:
						ax.plot_wireframe(*j(10), color = 'b')
				elif isinstance(j, Vector):
					ax.plot(*j.euclidean.T, color = 'b')
				else:
					raise ValueError("CamKII should return Objects")


		# ax.plot(*np.array(brow).T)


	# points = sphere.random(100, -np.pi/4)
	# if points != []:
	# 	ax.scatter(*np.array(list(points)).T, color = 'r')

	# points = sphere.random(100, np.pi/4)
	# print(points)
	# if points != []:
	# 	ax.scatter(*np.array(points).T, color = 'g')

	# points = sphere.random(100, 0)
	# print(points)
	# if points != []:
	# 	ax.scatter(*np.array(points).T, color = 'b')

	# sys.exit()

	# # brownian motion
	# size = 95
	# # np.random.seed(0)
	# brow = list(brownian2(D = 2, T = size, dt = 1, d = 3, origin = [0,0,0]))
	# brow2 = list(brownian2(D = 2, T = size, dt = 1, d = 3, origin = [20,20,0]))

	# for i in range(1, size):
	# 	v = Vector(brow[i], brow[i-1])
	# 	v2 = Vector(brow2[i], brow2[i-1])
	# 	camkii = CamKII(6, 1, 2, origin = brow2[i-1])
		
	# 	sphere = None
	# 	dist = []
	# 	b = False

	# 	for j in camkii:
	# 		if isinstance(j, HyperSphere):
	# 			tp, inters = j.intersection(v)
	# 			print(i, tp)
	# 			if tp == Intersection.segment_partial or tp == Intersection.segment_full:
	# 				print(tp, inters)
	# 				if tp == Intersection.segment_full:
	# 					dist.append((v.distance(inters[0], on = True), j))
	# 					dist.append((v.distance(inters[1], on = True), j))
	# 				else:
	# 					dist.append((v.distance(inters, on = True), j))
	# 				b = True
	# 		elif isinstance(j, Vector):
	# 			continue
	# 		else:
	# 			raise ValueError("CamKII should return Objects")

	# 	if b == True:
	# 		min_d = min([d for d,i in dist])
	# 		s = [i for d,i in dist if d == min_d][0]
	# 		print("b == True")
	# 		for j in camkii:
	# 			if isinstance(j, HyperSphere):
	# 				tp, inters = j.intersection(v)
	# 				if tp == Intersection.segment_partial or tp == Intersection.segment_full:
	# 					ax.plot_wireframe(*j(10), color = 'b')
	# 					if j == s:
	# 						ax.plot_wireframe(*s(10), color = 'c')
	# 						if tp == Intersection.segment_full:
	# 							if v.distance(inters[0], on = True) == min_d:
	# 								v = Vector(inters[0], v.origin)
	# 								ax.scatter(*inters[0].euclidean.T, color = 'r')
	# 							else:
	# 								v = Vector(inters[1], v.origin)
	# 								ax.scatter(*inters[1].euclidean.T, color = 'r')
	# 						else:
	# 							v = Vector(inters, v.origin)
	# 							ax.scatter(*inters.euclidean.T, color = 'r')
	# 						ax.plot(*v.euclidean.T, color = "k")
	# 						ax.scatter(*v.euclidean.T, color = "m")
	# 				else:
	# 					ax.plot_wireframe(*j(10), color = 'b')
	# 			elif isinstance(j, Vector):
	# 				ax.plot(*j.euclidean.T, color = 'b')
	# 			else:
	# 				raise ValueError("CamKII should return Objects")
	# 		break

	# 	ax.plot(*v.euclidean.T, color = "k")
	# 	ax.scatter(*v.euclidean.T, color = "m")
	# 	ax.plot(*v2.euclidean.T, color = "y")

	# if b == False:
	# 	for j in camkii:
	# 		if isinstance(j, HyperSphere):
	# 			ax.plot_wireframe(*j(10), color = 'b')
	# 		elif isinstance(j, Vector):
	# 			ax.plot(*j.euclidean.T, color = 'b')
	# 		else:
	# 			raise ValueError("CamKII should return Objects")

	ax.legend()

	# plt.show()

	# forloop used with %matplotlib notebook for interaction with the figure
	for angle in range(0, 360):
		ax.view_init(30, angle)
		plt.draw()
		plt.pause(.001)




