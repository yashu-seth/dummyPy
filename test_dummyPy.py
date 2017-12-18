import unittest

import numpy as np
import pandas as pd
import numpy.testing as np_test
from scipy.sparse import coo_matrix

from dummyPy import Encoder, OneHotEncoder


class TestEncoder(unittest.TestCase):
	def test_class(self):
		encoder = Encoder()
		self.assertEqual(encoder.column_mapper, None)

		levels = set()
		for color in ["red", "blue", "yellow"]:
			levels.add(color)

		encoder.fit(levels)
		self.assertEqual(encoder.column_mapper, {'blue': 0, 'red': 1, 'yellow': 2})

		data = pd.Series(["red", "red", "blue", "yellow", "brown", "red"])

		transformed_data1 = encoder.transform(data)
		transformed_data2 =  coo_matrix((np.ones(5), ([0, 1, 2, 3, 5], [1, 1, 0, 2, 1])),
					  					 shape=(6, 3))

		np_test.assert_array_equal(transformed_data1.toarray(), transformed_data2.toarray())


