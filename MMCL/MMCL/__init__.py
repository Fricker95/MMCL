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

from MMCL.Transformations 		import *
from MMCL.MFPT 					import *

from MMCL.Utils 					import percentageDifference, writeCSV, readCSV
from MMCL.Brownian 				import brownian
from MMCL.HyperSpace 			import Point, Vector, Plane
from MMCL.HyperSphere 			import Intersection, HyperSphere
from MMCL.Polygon 				import Polygon
from MMCL.CamKII 				import CamKII


hv.extension('matplotlib')
warnings.filterwarnings('ignore') #temporary fix to suppress deprecation warnings we get with matplotlib

# %matplotlib notebook
