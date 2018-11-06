import pandas as pd
import numpy as np
import scipy
import warnings
from scipy.stats import chi2_contingency
import warnings
from . import preprocessing

class attr_rank(object):
	def __init__(self):
		pass

	@classmethod 
	def convert_to_float(cls, var):
		return preprocessing.convert_to_float(var)

	@classmethod 
	def contingency_table(cls, x, y, continuous_x = 10, continuous_y = 10):
		"""
		To do:
			Make a contingency table (numpy array) for future use. 
		Parameter:
			x (numpy array): first attribute
			y (numpy array): second attribute
			continuous_x (int): number of parts divided for x if x is continuous
			continuous_y (int): number of parts divided for y if y is continuous
		Return:
			table (np.array)
		"""
		x = cls.convert_to_float(x)
		y = cls.convert_to_float(y)
		continuous_n = [continuous_x, continuous_y]
		if np.array_equal(x, y):
			return np.array() 
		else:
			if (len(x) != len(y)):
				raise ValueError("x and y don't have equal length.")
			else:
				x = cls.cut_variable_nth(x, continuous_x)
				y = cls.cut_variable_nth(y, continuous_y)
				table = pd.crosstab(x, y)
				return table

	@staticmethod
	def cut_variable_nth(x, n):
		if (x.dtype == 'float64' or x.dtype == 'int64')and len(np.unique(x)) > n:
			x = np.array(x)
			var = np.array(np.repeat('nan', len(x)), dtype = 'object')
			points = [np.nanpercentile(x, 100/n * i) for i in range(n)]
			# print(points)
			for idx in range(n-1):
				var[[v for v in range(len(x)) if points[idx] <= x[v] < points[idx + 1]]] = "%.2f - %.2f"%(points[idx], points[idx + 1])
			return var
		else:
			return x
