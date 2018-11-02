from collections import defaultdict
from pickle import load, dump

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack


def sort_mixed(levels):
    try:
        return sorted(levels)
    except TypeError:
        str_list = [l for l in levels if isinstance(l, str)]
        other_list = [l for l in levels if isinstance(l, (int, float)) and not np.isnan(l)]
        nan_append = [l for l in levels if isinstance(l, (int, float)) and np.isnan(l)]
        return sorted(str_list) + sorted(other_list) + nan_append


class Encoder:
    """
    Helper class to encode levels of a categorical Variable.
    """
    def __init__(self):
        self.column_mapper = None

    def fit(self, levels):
        """
        Parameters
        ----------
        levels: set
            Unique levels of the categorical variable.
        """
        self.column_mapper = {x: i for i, x in enumerate(sort_mixed(levels))}

    def transform(self, column_data):
        """
        Parameters
        ----------
        column_data: pandas Series object
        """
        row_cols = [(i, self.column_mapper[x])
                    for i, x in enumerate(column_data) if x in self.column_mapper]
        data = np.ones(len(row_cols))

        if len(row_cols) == 0:
            return coo_matrix((column_data.shape[0], len(self.column_mapper)))

        return coo_matrix((data, zip(*row_cols)), shape=(column_data.shape[0], len(self.column_mapper)))

    def __eq__(self, other):
        return self.column_mapper == other.column_mapper


class OneHotEncoder:
    """
    A One Hot Encoder class that converts the categorical variables in a data frame
    to one hot encoded variables. It can also handle large data that is too big to fit
    in the memory by reading the data in chunks.

    Example
    -------
    The following example uses the kaggle's titanic data. It can be found here -
    `https://www.kaggle.com/c/titanic/data`

    This data is only 60 KB and it has been used for a demonstration purpose.
    This class also works well with datasets too large to fit into the machine
    memory.

    >>> from dummyPy import OneHotEncoder
    >>> import pandas as pd
    >>> encoder = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])
    >>> data = pd.read_csv("titanic.csv", usecols=["Pclass", "Sex", "Age", "Fare", "Embarked"])
    >>> data.shape
    (891, 5)
    >>> encoder.fit(data)
    >>> X = encoder.transform(data)
    >>> X.shape
    (891, 11)
    >>> X
    array([[0.0, 0.0, 1.0, ..., 0.0, 0.0, 1.0],
           [1.0, 0.0, 0.0, ..., 1.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, ..., 0.0, 0.0, 1.0],
           ...,
           [0.0, 0.0, 1.0, ..., 0.0, 0.0, 1.0],
           [1.0, 0.0, 0.0, ..., 1.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, ..., 0.0, 1.0, 0.0]], dtype=object)

    >>> chunked_data = pd.read_csv("titanic.csv",
                                    usecols=["Pclass", "Sex", "Age", "Fare", "Embarked"],
                                    chunksize=10)
    >>> encoder2 = OneHotEncoder(categorical_columns=["Pclass", "Sex", "Embarked"])
    >>> encoder2.fit(chunked_data)
    >>> X = encoder2.transform(data)
    >>> X.shape
    (891, 11)
    >>> X
    array([[0.0, 0.0, 1.0, ..., 0.0, 0.0, 1.0],
           [1.0, 0.0, 0.0, ..., 1.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, ..., 0.0, 0.0, 1.0],
           ...,
           [0.0, 0.0, 1.0, ..., 0.0, 0.0, 1.0],
           [1.0, 0.0, 0.0, ..., 1.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, ..., 0.0, 1.0, 0.0]], dtype=object)
    
    """
    def __init__(self, categorical_columns=None, file_name=None):
        """
        Parameters
        ----------

        categorical_columns: list
            A list of the names of the categorical varibales in the data. All these columns
            must have dtype as string.
        file_name: string
            The file name to load a saved encoder from.
        """
        if file_name is None:
            if categorical_columns is None:
                raise UserWarning("Either the categorical columns must be defined "
                                  "or a file_name for a saved encoder must be given")
            self.categorical_columns = categorical_columns
            self.unique_vals = defaultdict(set)
            self.encoders = {column_name: Encoder() for column_name in categorical_columns}
        elif isinstance(file_name, str):
            self.load(file_name)
        else:
            raise UserWarning("The file name to load a saved encoder should be a string to a readable file")

    def _update_unique_vals(self, data):
        for column_name in self.categorical_columns:
            for value in data[column_name]: 
            	self.unique_vals[column_name].add(value)

    def _fit_encoders(self):
        for column_name in self.categorical_columns:
            self.encoders[column_name].fit(self.unique_vals[column_name])

    def fit(self, data):    
        """
        This method reads the categorical columns and gets the necessary
        one hot encoded column shapes.
        It can also read the data in chunks.

        Parameters
        ----------
        data: pandas.core.frame.DataFrame or pandas.io.parsers.TextFileReader

            The data can be either a pandas data frame or a pandas TextFileReader
            object. The TextFileReader object is created by specifying the 
            chunksize parameter in pandas read_csv method.
        
            Use the TextFileReader object as input if the dataset is too large to
            fit in the machine memory.
        """
        if(isinstance(data, pd.core.frame.DataFrame)):
            self._update_unique_vals(data)
        else:
            for data_chunk in data:
                self._update_unique_vals(data_chunk)

        self._fit_encoders() 

    def transform(self, data, dtype="pd"):
        """
        This method is used to convert the categorical values in your data into
        one hot encoded vectors. It convets the categorical columns in the data
        to one hot encoded columns and leaves the continuous variable columns as it is.

        Parameters
        ----------
        data: pandas data frame
            The data frame object that needs to be transformed.

        dtype: string 
            "pd"  - This will return a pandas dataframe.
            "np"  - This will return a numpy array.
            "coo" - This will rerurn scipy.sparse.coo_matrix, which is memory-efficient 
                    for categorical variable of which number of unique values are large.
        """
        transformed_coo_matrix = hstack([self.encoders[column_name].transform(data[column_name])
                                         if column_name in self.categorical_columns
                                         else coo_matrix(data[column_name].values.reshape(-1, 1))
                                         for column_name in data.columns])

        if dtype == "np":
            return transformed_coo_matrix.toarray()
        elif dtype == "coo":
            return transformed_coo_matrix
        else:

            # For the titanic example, the Nested List mentioned below would look like -
            # [["Pclass_0", "Pclass_1", "Pclass_2"], ["Sex_female", "Sex_male"], ["Age"], ["Fare"],
            #  ["Embarked_Q", "Embarked_nan", "Embarked_S", "Embarked_C"]]
            # It is flattened later.

            transformed_data_col_names = [item for sublist in 
                                          # Nested List
                                          [[column_name] if column_name not in self.categorical_columns
                                           else ["{}_{}".format(column_name, x)
                                                 for x in sort_mixed(self.unique_vals[column_name])]
                                           for column_name in data.columns]
                                          for item in sublist]
            return pd.DataFrame(transformed_coo_matrix.toarray(), columns=transformed_data_col_names)

    def fit_transform(self, data):
        """
        This method calls fit and transform one after the other.
        
        Please note that unlike the fit method the fit_transform method
        can take only the pandas data frame as input. 

        Parameters
        ----------
        data: pandas.core.frame.DataFrame
            A pandas data frame.
        """
        self.fit(data)
        return self.transform(data)

    def save(self, file_name):
        """
        Saves the encoder as a pickled binary file with the path file_name
        :param file_name: string
            The file name to use for the saved encoder.
        :return:
        """
        with open(file_name, "wb") as file:
            d = {
                "categorical_columns": self.categorical_columns,
                "unique_vals": self.unique_vals,
                "encoders": self.encoders
            }
            dump(d, file)

    def load(self, file_name):
        """
        Loads a pickled encoder from a file with the path file_name
        :param file_name: string
            The file name to load the saved encoder from.
        :return:
        """
        with open(file_name, "rb") as file:
            d = load(file)
            self.categorical_columns = d["categorical_columns"]
            self.unique_vals = d["unique_vals"]
            self.encoders = d["encoders"]
