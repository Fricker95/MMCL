#! /bin/python3

"""
Utils.py

Created by Nicolas Fricker on 1/1/21.
Copyright © 2021 Nicolas Fricker. All rights reserved.
"""

import numpy as np
import csv
import sys

CIRCLE_DEGREES 		= 360
HALFCIRLCE_DEGREES 	= 180

def isBetween(a, b, c, accuracy = 16):
	# a |<------>| b |<------>| c
	ac = np.linalg.norm(c - a)
	ab = np.linalg.norm(b - a)
	bc = np.linalg.norm(c - b)
	
	return np.round(ab + bc, accuracy) == np.round(ac, accuracy)

def degrees2Radians(theta):
	return theta * np.pi / HALFCIRLCE_DEGREES

def radians2Degrees(rad):
	return rad * HALFCIRLCE_DEGREES / np.pi

def coordAngle(xy, R = 1, origin = (0, 0)):
	angle = radians2Degrees(np.arctan2(xy[1] - origin[1], xy[0] - origin[0]))
	if angle < 0:
		return CIRCLE_DEGREES + angle
	return angle

def normalize(x, a, b):
	return (b - a) * ((x - min(x)) / (max(x) - min(x))) + a

def roundDown(n, decimals=0): 
    multiplier = 10 ** decimals 
    return np.floor(n * multiplier) / multiplier 

def percentageDifference(v1, v2):
	return abs((v1 - v2)/((v1 + v2)/2)) * 100

def writeCSV(fdir, data, headers = [], writeheader = False):
	if len(headers) > 0:
		with open(fdir, "a") as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames = headers)
			if writeheader:
				writer.writeheader()
			if len(data) > 0:
				writer.writerows(data)

def readCSV(fdir):
	data = []
	with open(fdir, "r") as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			data.append(row)
	return data



	