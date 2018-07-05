import unittest
from collections import defaultdict

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
        self.assertEqual(encoder.column_mapper,
                         {'blue': 0, 'red': 1, 'yellow': 2})

        data = pd.Series(["red", "red", "blue", "yellow", "brown", "red"])

        transformed_data1 = encoder.transform(data)
        transformed_data2 = coo_matrix((np.ones(5),
                                       ([0, 1, 2, 3, 5], [1, 1, 0, 2, 1])),
                                       shape=(6, 3))

        np_test.assert_array_equal(transformed_data1.toarray(),
                                   transformed_data2.toarray())


class TestOneHotEncoder(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv("titanic.csv",
                                usecols=["Pclass", "Sex", "Age", "Fare", "Embarked"])
        self.chunked_data = pd.read_csv("titanic.csv",
                                        usecols=["Pclass", "Sex", "Age", "Fare", "Embarked"],
                                        chunksize=10)

    def test_class_init(self):
        one_hot_encoder = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])

        self.assertEqual(one_hot_encoder.categorical_columns, ["Pclass", "Sex", "Embarked"])
        self.assertEqual(one_hot_encoder.unique_vals, defaultdict(set))
        self.assertEqual(one_hot_encoder.encoders,
                         {"Pclass": Encoder(), "Sex": Encoder(), "Embarked": Encoder()})

    def test_update_unique_vals(self):
        one_hot_encoder = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])
        one_hot_encoder._update_unique_vals(self.data)

        self.assertEqual(one_hot_encoder.unique_vals["Embarked"], set(['Q', np.nan, 'S', 'C']))
        self.assertEqual(one_hot_encoder.unique_vals["Sex"], set(['male', 'female']))
        self.assertEqual(one_hot_encoder.unique_vals["Pclass"], set([1, 2, 3]))

    def test_fit_encoders(self):
        one_hot_encoder = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])
        one_hot_encoder._update_unique_vals(self.data)
        one_hot_encoder._fit_encoders()

        embarked_encoder = Encoder()
        embarked_encoder.fit(set(['Q', np.nan, 'S', 'C']))
        self.assertEqual(one_hot_encoder.encoders["Embarked"], embarked_encoder)

        sex_encoder = Encoder()
        sex_encoder.fit(set(['male', 'female']))
        self.assertEqual(one_hot_encoder.encoders["Sex"], sex_encoder)

        pclass_encoder = Encoder()
        pclass_encoder.fit(set([1, 2, 3]))
        self.assertEqual(one_hot_encoder.encoders["Pclass"], pclass_encoder)

    def test_fit(self):
        one_hot_encoder1 = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])
        one_hot_encoder2 = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])

        one_hot_encoder1.fit(self.data)
        one_hot_encoder2._update_unique_vals(self.data)
        one_hot_encoder2._fit_encoders()

        self.assertEqual(one_hot_encoder1.categorical_columns,
                         one_hot_encoder2.categorical_columns)
        self.assertEqual(one_hot_encoder1.unique_vals,
                         one_hot_encoder2.unique_vals)
        self.assertEqual(one_hot_encoder1.encoders,
                         one_hot_encoder2.encoders)

    def test_fit_chunks(self):
        one_hot_encoder1 = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])
        one_hot_encoder2 = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])

        one_hot_encoder1.fit(self.chunked_data)
        one_hot_encoder2.fit(self.data)

        self.assertEqual(one_hot_encoder1.categorical_columns,
                         one_hot_encoder2.categorical_columns)
        self.assertEqual(one_hot_encoder1.unique_vals,
                         one_hot_encoder2.unique_vals)
        self.assertEqual(one_hot_encoder1.encoders,
                         one_hot_encoder2.encoders)

    def test_transform(self):
        one_hot_encoder = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])
        one_hot_encoder.fit(self.data)

        transformed_data = np.array([[0.0, 0.0, 1.0, 0.0, 1.0, 22.0, 7.25, 0.0, 0.0, 0.0, 1.0],
                                     [1.0, 0.0, 0.0, 1.0, 0.0, 38.0, 71.2833, 0.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 1.0, 0.0, 26.0, 7.925, 0.0, 0.0, 0.0, 1.0],
                                     [1.0, 0.0, 0.0, 1.0, 0.0, 35.0, 53.1, 0.0, 0.0, 0.0, 1.0],
                                     [0.0, 0.0, 1.0, 0.0, 1.0, 35.0, 8.05, 0.0, 0.0, 0.0, 1.0]])

        np_test.assert_array_equal(one_hot_encoder.transform(self.data.head()),
                                   transformed_data)
    def test_transform_coo(self):
        one_hot_encoder = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])
        one_hot_encoder.fit(self.data)
        coo_matrix_1 = one_hot_encoder.transform(self.data.head(), dtype="coo")
        coo_matrix_2 = coo_matrix(one_hot_encoder.transform(self.data.head(), dtype="np"))
        np_test.assert_array_equal(coo_matrix_1.toarray(), 
                                   coo_matrix_2.toarray())

    def test_fit_transform(self):
        one_hot_encoder1 = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])
        one_hot_encoder2 = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])

        one_hot_encoder2.fit(self.data.head())

        np_test.assert_array_equal(one_hot_encoder1.fit_transform(self.data.head()),
                                   one_hot_encoder2.transform(self.data.head()))

