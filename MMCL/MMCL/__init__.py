#! /bin/python3

"""
__init__.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.

Run: 
	pip install -e <dir path to package>
"""

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

from pprint 				import pprint

from Transformations 		import *
from MFPT 					import *

from Utils 					import percentageDifference, writeCSV, readCSV
from Brownian 				import brownian
from HyperSpace 			import Point, Vector, Plane
from HyperSphere 			import Intersection, HyperSphere
from Polygon 				import Polygon
from CamKII 				import CamKII


hv.extension('matplotlib')
warnings.filterwarnings('ignore') #temporary fix to suppress deprecation warnings we get with matplotlib

# %matplotlib notebook
