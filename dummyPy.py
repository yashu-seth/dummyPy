from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


class OneHotEncoder():
	"""
	A One Hot Encoder class that converts the categorical variables in a data frame
	to one hot encoded variables. It can also handle large data that is too big to fit
	in the memory by reading the the data in chunks.

	Example
	-------
	The following example uses the kaggle's titanic data. It can found here -
	`https://www.kaggle.com/c/titanic/data`

	This data is only 60 KB and it has only been used for a demonstration purpose.
	The actual use of this class is for datasets that cannot fit into memory.  

	>>> from dummypy import OneHotEncoder
	>>> import pandas as pd
	>>> encoder = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])
	>>> encoder.fit(file_path="titanic.csv", chunksize=100,
					usecols=["Pclass", "Sex", "Age", "Fare", "Embarked"], na_values=str)
	>>> data = pd.read_csv("titanic.csv", usecols=["Pclass", "Sex", "Age", "Fare", "Embarked"])
	>>> X = encoder.transform(data)
	>>> X

	array([[ 0.,  0.,  1., ...,  0.,  0.,  1.],
	       [ 1.,  0.,  0., ...,  1.,  0.,  0.],
	       [ 0.,  0.,  1., ...,  0.,  0.,  1.],
	       ..., 
	       [ 0.,  0.,  1., ...,  0.,  0.,  1.],
	       [ 1.,  0.,  0., ...,  1.,  0.,  0.],
	       [ 0.,  0.,  1., ...,  0.,  1.,  0.]])
	"""
	def __init__(self, categorical_columns):
		"""
		Parameters
		----------

		categorical_columns: list
			A list of the names of the categorical varibales in the data. All these columns
			must have dtype as string.
		"""

		self.categorical_columns = categorical_columns
		self.unique_vals = defaultdict(set)
		self.encoders = {column_name : LabelBinarizer() for column_name in categorical_columns}

	def _update_unique_vals(self, data_chunk):
		for column_name in self.categorical_columns:
			for value in data_chunk[column_name]: self.unique_vals[column_name].add(value)

	def _fit_encoders(self):
		for column_name in self.categorical_columns:
			self.encoders[column_name].fit(list(self.unique_vals[column_name]))

	def fit(self, file_path, chunksize=1000, **kwargs):	
		"""
		This method reads the categorical columns in chunks and gets the necessary
		one hot encoded columns shapes.

		Parameters
		----------

		file_path: string
			The file path of the data that is to be read in chunks.

		chunksize: integer, default = 1000
			The size of the invidual chunk which have to be read all
			at once in the memory. Care must be taken to ensure that the size
			of an individual chunk does not exceed the available memory.
			For details, refer chunksize parameter of pandas.read_csv.

		kwargs:
			The other parameters of the read_csv method in pandas.
			For example, you can use the usecols parameter to read only particular columns
			of your data. 

		"""

		if(set(self.categorical_columns) - set(pd.read_csv(file_path, nrows=1, **kwargs).columns)):
			raise ValueError("One or more of the specified categorical columns "\
							  "is not present in the given data.")

		data = pd.read_csv(file_path, chunksize=chunksize, **kwargs)

		for data_chunk in data:
			self._update_unique_vals(data_chunk)

		self._fit_encoders()

	def transform(self, data):
		"""
		This method is used to convert the categorical values in your data into
		one hot encoded vectors. It convets the categorical columns in the data
		to one hot encoded columns and leaves the continuous variable cloumns as it is.

		Parameters
		----------
		data: pandas data frame
			The data frame obhect that needs to be transformed.
		"""
		transformed_data = [self.encoders[column_name].transform(data[column_name]) 
						    if column_name in self.categorical_columns
						    else data[column_name].reshape(-1, 1)
						    for column_name in data.columns]
		return(np.concatenate(transformed_data, axis=1))
