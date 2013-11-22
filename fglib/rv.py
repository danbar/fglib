"""Module for random variables.

This module contains classes for different random variables.

Classes:
Discrete -- Discrete random variable
Gaussian -- Gaussian random variable

"""

import numpy as np


class Discrete(object):

    """Discrete random variable.

    The discrete random variable is interally implemented in ...

    """

    index = None  # Static index variable
    state = None  # Static state variable

    def __init__(self, value, index):
        """Discrete random variable.

        Global variables 'index', 'state' have to be defined prior.
        Value is an array with probabilities and
        index a dictionary of variable node to index pairs.

        """
        assert Discrete.index is not None
        assert Discrete.state is not None
        self.value = value
        self.index = index

        self.vars = list(index.keys())  # Variables
        self.dims = list(index.values())  # Dimensions (list of indices)

    def __str__(self):
        """Return string representation."""
        return str(self.value)

    def __add__(self, other):
        """Perform addition and return the result."""
        (x, y) = Discrete.__expand(self, other)
        return Discrete(x + y, self.index)

    def __mul__(self, other):
        """Perform multiplication and return the result."""
        (x, y) = Discrete.__expand(self, other)
        return Discrete(x * y, self.index)

    @staticmethod
    def __expand(a, b):
        """Adjust dimensions of to discrete random variables and return it.

        TODO: ...

        """
        x = a.value
        y = b.value

        # Case: x = vector, y = matrix
        if x.ndim < y.ndim:
            i = b.index[a.vars[0]]  # Vector dimension in matrix coordinates

            for d in sorted(b.dims):  # lowest dimension to highest dimension
                if d < i:
                    x = np.repeat(x[np.newaxis, :], len(Discrete.state),
                                  axis=0)  # Prepend dimension
                elif d > i:
                    x = np.repeat(x[:, np.newaxis], len(Discrete.state),
                                  axis=1)  # Append dimension

        # Case: x = matrix, y = vector
        elif x.ndim > y.ndim:
            i = a.index[b.vars[0]]

            for d in sorted(a.dims):
                if d < i:
                    y = np.repeat(y[np.newaxis, :], len(Discrete.state),
                                  axis=0)
                elif d > i:
                    y = np.repeat(y[:, np.newaxis], len(Discrete.state),
                                  axis=1)

        return x, y  # Return discrete random variables with equal dimensions

    def __iadd__(self, other):
        """Method for augmented addition and return the result."""
        return self.__add__(other)

    def __imul__(self, other):
        """Method for augmented multiplication and return the result."""
        return self.__mul__(other)

    def argmax(self, var):
        """Return the index of the maximum in dimension of variable."""
        return np.argmax(self.value, self.index[var])

    def int(self, var):
        """Perform summation of specific dimension and return the result."""
        val = np.sum(self.value, self.index[var])
        return Discrete(val, \
                        {k: v for k, v in self.index.items() if k is not var})

    def log(self):
        """Return the natural logarithm of the random variable."""
        return Discrete(np.log(self.value), self.index)

    def max(self, var):
        """Perform maximization of specific dimension and return the result."""
        val = np.amax(self.value, self.index[var])
        return Discrete(val, \
                        {k: v for k, v in self.index.items() if k is not var})


class Gaussian(object):

    """Gaussian random variable.

    The Gaussian random variable is internally implemented in information form.

    """

    def __init__(self, mean=[[0]], cov=[[1]]):
        """Gaussian random variable.

        Create a Gaussian random variable with corresponding
        mean and covariance matrix.

        """
        self.W = np.linalg.inv(np.asarray(cov))
        self.Wm = np.dot(self.W, np.asarray(mean))
        self.ndim = self.W.shape[0]

    @classmethod
    def moment_form(cls, m, V):
        """Return a Gaussian random variable from a given moment form."""
        return Gaussian(m, V)

    @classmethod
    def information_form(cls, W, Wm):
        """Return a Gaussian random variable from a given information form."""
        self = Gaussian()
        self.W = np.asarray(W)
        self.Wm = np.asarray(Wm)
        self.ndim = self.W.shape[0]
        return self

    @property
    def mean(self):
        """Return the mean."""
        return np.dot(np.linalg.inv(self.W), self.Wm)

    @property
    def cov(self):
        """Return the covariance matrix."""
        return np.linalg.inv(self.W)

    def __str__(self):
        """Return string representation."""
        return "%s %s" % (self.mean, self.cov)

    def __add__(self, other):
        """Perform addition and return the result."""
        return Gaussian(self.mean + other.mean,
                        self.cov + other.cov)

    def __sub__(self, other):
        """Perform subtraction and return the result."""
        return Gaussian(self.mean - other.mean,
                        self.cov - other.cov)

    def __mul__(self, other):
        """Perform multiplication and return the result."""
        W = self.W + other.W
        Wm = self.Wm + other.Wm
        return Gaussian.information_form(W, Wm)

    def __iadd__(self, other):
        """Perform augmented addition and return the result."""
        return self.__add__(other)

    def __isub__(self, other):
        """Perform augmented subtraction and return the result."""
        return self.__sub__(other)

    def __imul__(self, other):
        """Perform augmented multiplication and return the result."""
        return self.__mul__(other)

    def __eq__(self, other):
        """Perform comparison and return the boolean result."""
        return np.allclose(self.W, other.W) \
            and np.allclose(self.Wm, other.Wm)

    def argmax(self, dim=None):
        """Return the argument of the maximum for a given dimension."""
        if dim is None:
            return self.mean
        return self.mean[np.ix_(dim, [0])]

    def max(self, dim=None):
        """Return the maximum for a given dimension."""
        if dim is None:
            return np.power(2 * np.pi, self.ndim / 2) * \
                np.sqrt(np.linalg.det(self.cov))
        g = self.int(dim)
        return np.power(2 * np.pi, g.ndim / 2) * np.sqrt(np.linalg.det(g.cov))

    def int(self, dim):
        """Return the marginal for a given dimension."""
        return Gaussian(self.mean[np.ix_(dim, [0])],
                        self.cov[np.ix_(dim, dim)])

    def log(self):
        """Return the natural logarithm of the random variable."""
        # TODO: Not implemented for Gaussian random variable.
        raise NotImplementedError
