#! /bin/python3

"""
main.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.
"""

from Transformations 		import *
from MFPT 					import *

from Utils 					import percentageDifference, writeCSV, readCSV
from Brownian 				import brownian
from HyperSpace 			import Point, Vector, Plane
from HyperSphere 			import Intersection, HyperSphere
from Polygon 				import Polygon
from CamKII 				import CamKII

################# IN TESTING #################

from Simulations 			import *

##############################################


from matplotlib 			import pyplot as plt
# import matplotlib.animation as animation
from mpl_toolkits.mplot3d 	import Axes3D

from multiprocessing 		import Process, Queue

import multiprocessing as mp
import holoviews as hv
import numpy as np
import warnings
import time
import csv
import sys

hv.extension('matplotlib')
warnings.filterwarnings('ignore') #temporary fix to suppress deprecation warnings we get with matplotlib

# %matplotlib notebook

# np.random.seed(0)	

def plot_sphere_in_sphere_data(fdir, title):
	fig, axs = plt.subplots(2, 2, figsize = (10, 10))

	data = readCSV(fdir)

	plot_data = {k: [] for k in data[0].keys()}

	for i in data:
		print(i)
		for k,v in i.items():
			plot_data[k].append(float(v))

	print(plot_data)

	xaxis = ['Inner Sphere Radius (R1)', 'Diffusion Coefficient (D)', 'Delta t (∆t)', 'N Simulations']
	ttype = ['R1', 'D', 'dt', 'N_sim']


	for i in range(2):
		for j in range(2):
			axs[i, j].set(xlabel = xaxis[i*2+j])
			if j == 0:
				axs[i, j].set(ylabel = "MFPT")
			axs[i, j].grid(True)
			axs[i, j].set_yscale('log')

			index = (i*2+j) * 3
			print("type", ttype[i*2+j])
			print("data",plot_data[ttype[i*2+j]][index: index+3])
			axs[i, j].plot(plot_data[ttype[i*2+j]][index: index+3], plot_data["anali_time"][index: index+3], color='red', marker='o', label = "Analytical mfpt")

			axs[i, j].plot(plot_data[ttype[i*2+j]][index: index+3], plot_data["sim_time"][index: index+3], color='blue', marker='o', label = "Simulation mfpt")

			axs[i, j].legend()

	fig.suptitle(title)
	plt.show()

def simulations_sphere_in_sphere(dim = 3, n_threads = mp.cpu_count() - 1):
	if dim == 2:
		fdir = "../bkup/result2D.csv"
	if dim == 3:
		fdir = "../bkup/result3D.csv"

	headers = ["D", "R1", "R2", "dt", "N_sim", "d", "exec_time", "sim_time", "anali_time", "percent_diff", "failed"]

	data = []
	writeCSV(fdir, data, headers, True)

	print("started")

	for i in [10, 20, 30]:
		data.append(simulation_sphere_in_sphere_threadpool(D = 1, R1 = i, R2 = 100, dt = 1, n_threads = n_threads, N_sim = 1e3, dim = dim))

	writeCSV(fdir, data, headers)
	data = []

	print("R1 finished")

	for i in [1, 5, 10]:
		data.append(simulation_sphere_in_sphere_threadpool(D = i, R1 = 10, R2 = 100, dt = 1, n_threads = n_threads, N_sim = 1e3, dim = dim))


	writeCSV(fdir, data, headers)
	data = []

	print("D finished")

	for i in [0.1, 0.5, 1]:
		data.append(simulation_sphere_in_sphere_threadpool(D = 1, R1 = 10, R2 = 100, dt = i, n_threads = n_threads, N_sim = 1e3, dim = dim))

	writeCSV(fdir, data, headers)
	data = []

	print("dt finished")


	for i in [1e2, 1e3, 1e4]:
		data.append(simulation_sphere_in_sphere_threadpool(D = 1, R1 = 10, R2 = 100, dt = 1, n_threads = n_threads, N_sim = i, dim = dim))


	writeCSV(fdir, data, headers)
	data = []

	print("N_sim finished")

	print("done")

	plot_sphere_in_sphere_data(fdir, f"MFPT {dim}D")


def main():
	# plot_sphere_in_sphere_data("../bkup/result2D.csv", f"MFPT {2}D")
	# simulations_sphere_in_sphere(2)
	# simulations_sphere_in_sphere(3)

	simplified_simulation_sphere_in_sphere_plot(1, 10, 100, 2, 1, 1e9 )
	# simplified_simulation_sphere_in_sphere_plot(10, 10, 100, 3, 1, 1e9)

	# simplified_simulation_CAMKII_1_sphere_in_sphere_plot(1, 100, 2, 1, 1e6)
	# simplified_simulation_CAMKII_1_sphere_in_sphere_plot(1, 100, 3, 1, 1e6)

	# simplified_simulation_CAMKII_2_sphere_in_sphere_plot(1, 100, 2, 1, 1e6)
	# simplified_simulation_CAMKII_2_sphere_in_sphere_plot(1, 100, 3, 1, 1e6)

	# simplified_simulation_CAMKII_3_sphere_in_sphere_plot(1, 100, 2, 1, 1e6)
	# simplified_simulation_CAMKII_3_sphere_in_sphere_plot(1, 100, 3, 1, 1e6)

	# simplified_simulation_CAMKII_4_sphere_in_sphere_plot(1, 100, 2, 1, 1e6)
	# simplified_simulation_CAMKII_4_sphere_in_sphere_plot(1, 100, 3, 1, 1e6)


	# real_simulation_plot()

	pass


if __name__ == '__main__':
	main()


