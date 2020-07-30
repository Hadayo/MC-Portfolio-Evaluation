import numpy as np


class ExpandArray(object):
    """A 2D array class with a fixed ahead number of rows and and dynamically
    expanding number of columns. The data itself is stored as a nested python
    list and is converted to numpy array when the user wants to.

    Parameters
    ----------
    num_rows : int
        Number of rows in the array.

    Attributes
    ----------
    data : list
        A 2D nested list containing the 2D array.
    is_empty : bool
        Signifies if the array wasn't filled yet.
    num_rows

    """
    def __init__(self, num_rows):
        self.num_rows = num_rows
        self.is_empty = True
        self.data = [[] for i in range(num_rows)]

    def append_col(self, col):
        """Appends a column to the end of the array.

        Parameters
        ----------
        col : array-like
            A 1 d array or a list of length num_rows containing the column.

        """
        err_msg = f"Column length ({len(col)}) must match num_rows ({self.num_rows})"
        assert len(col) == self.num_rows, err_msg

        for i, row in enumerate(self.data):
            row.append(col[i])
        self.is_empty = False

    def as_array(self):
        """Return the data as an ndarray.

        Returns
        -------
        ndarray

        """
        if self.num_rows == 1:
            return np.array(self.data).reshape(-1)  # return 1D vector
        return np.array(self.data)

    def __str__(self):
        return self.as_array().__str__()

    def last_col(self):
        """Return the last column of the array. If the array has one row, then
        the function returns its last element.

        Returns
        -------
        list or float

        """
        if self.num_rows == 1:
            return self.data[0][-1]
        else:
            return [row[-1] for row in self.data]
