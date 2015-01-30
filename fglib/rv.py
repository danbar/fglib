"""Module for random variables.

This module contains classes for random variables.

Classes:
Discrete -- Discrete random variable
Gaussian -- Gaussian random variable

"""

import numpy as np


class ParameterException(Exception):

    """Exception for invalid parameters."""

    pass


class Discrete(object):

    """Discrete random variable.

    Discrete random variables are internally implemented using
    multidimensional arrays for storing the probability mass function.

    """

    def __init__(self, raw_pmf, *args):
        """Discrete random variable.

        Create a discrete random variable with corresponding
        probability mass function.

        Positional arguments:
        pmf -- probability mass function
        dim -- dimensions

        """
        pmf = np.asarray(raw_pmf, dtype=np.float64)

        # Set probability mass function
        if not np.allclose(np.sum(pmf), 1):
            raise ParameterException('Invalid probability mass function.')
        else:
            self.pmf = pmf

        # Set variable nodes for dimensions
        if np.ndim(pmf) != len(args):
            raise ParameterException('Dimension mismatch.')
        else:
            self.dim = args

    def __str__(self):
        """Return string representation."""
        return str(self.pmf)

    def __add__(self, other):
        """Add other to self and return the result."""
        pmf = np.convolve(self.pmf, other.pmf, 'same')
        return Discrete(pmf, self.dim)

    def __sub__(self, other):
        """Subtract other from self and return the result."""
        pmf = np.convolve(self.pmf[::-1], other.pmf, 'same')
        return Discrete(pmf, self.dim)

    def __mul__(self, other):
        """Multiply other with self and return the result."""
        if len(self.dim) < len(other.dim):
            self._expand(other.dim)
        elif len(self.dim) > len(other.dim):
            other._expand(self.dim)

        pmf = self.pmf * other.pmf
        pmf /= np.sum(pmf)
        return Discrete(pmf, *self.dim)

    def __iadd__(self, other):
        """Method for augmented addition."""
        return self.__add__(other)

    def __isub__(self, other):
        """Method for augmented subtraction."""
        return self.__sub__(other)

    def __imul__(self, other):
        """Method for augmented multiplication."""
        return self.__mul__(other)

    def __eq__(self, other):
        """Compare self with other and return the boolean result."""
        return np.allclose(self.pmf, other.pmf) \
            and self.dim == other.dim

    def _expand(self, dims):
        """Expand dimensions.

        Expand the discrete random variable with the given new dimensions.

        Positional arguments:
        dims -- new dimensions

        """
        reps = [1, ] * len(dims)

        # Extract missing dimensions
        diff = [i for i, d in enumerate(dims) if d not in self.dim]

        # Expand missing dimensions
        for d in diff:
            self.pmf = np.expand_dims(self.pmf, axis=d)
            reps[d] = 2

        # Repeat missing dimensions
        self.pmf = np.tile(self.pmf, reps)
        self.dim = dims

    def marginal(self, *dims):
        """Return the marginal for a given dimension."""
        axis = tuple(idx for idx, d in enumerate(self.dim) if d in dims)
        pmf = np.sum(self.pmf, axis)
        pmf /= np.sum(pmf)

        new_dims = tuple(d for d in self.dim if d not in dims)
        return Discrete(pmf, *new_dims)

    def argmax(self, dim=None):
        """Return the argument of the maximum for a given dimension."""
        if dim is None:
            return np.unravel_index(self.pmf.argmax(), self.pmf.shape)
        m = self.marginal(dim)
        return np.argmax(m.pmf)

    def max(self, dim=None):
        """Return the maximum for a given dimension."""
        if dim is None:
            return np.amax(self.pmf)
        m = self.marginal(dim)
        return np.amax(m.pmf)

    def log(self):
        """Return the natural logarithm of the random variable."""
        return Discrete(np.log(self.value), self.index)


class Gaussian(object):

    """Gaussian random variable.

    Gaussian random variable are internally implemented using
    the information form for storing the probability density function.

    """

    def __init__(self, mean=[[0]], cov=[[1]]):
        """Gaussian random variable.

        Create a Gaussian random variable with corresponding
        mean and covariance matrix.

        Keyword arguments:
        mean -- mean vector
        cov -- covariance matrix

        """
        self.W = np.linalg.inv(np.asarray(cov))  # Precision matrix
        self.Wm = np.dot(self.W, np.asarray(mean))  # Precision-mean vector
        self.ndim = self.W.shape[0]  # Number of dimensions

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
    def m(self):
        """Return the mean vector."""
        return np.dot(np.linalg.inv(self.W), self.Wm)

    @property
    def V(self):
        """Return the covariance matrix."""
        return np.linalg.inv(self.W)

    @property
    def mean(self):
        """Return the mean vector."""
        return self.m

    @property
    def cov(self):
        """Return the covariance matrix."""
        return self.V

    def __str__(self):
        """Return string representation."""
        return "%s %s" % (self.mean, self.cov)

    def __add__(self, other):
        """Add other to self and return the result."""
        return Gaussian(self.mean + other.mean,
                        self.cov + other.cov)

    def __sub__(self, other):
        """Subtract other from self and return the result."""
        return Gaussian(self.mean - other.mean,
                        self.cov - other.cov)

    def __mul__(self, other):
        """Multiply other with self and return the result."""
        W = self.W + other.W
        Wm = self.Wm + other.Wm
        return Gaussian.information_form(W, Wm)

    def __iadd__(self, other):
        """Method for augmented addition."""
        return self.__add__(other)

    def __isub__(self, other):
        """Method for augmented subtraction."""
        return self.__sub__(other)

    def __imul__(self, other):
        """Method for augmented multiplication."""
        return self.__mul__(other)

    def __eq__(self, other):
        """Compare self with other and return the boolean result."""
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
        m = self.marginal(dim)
        return np.power(2 * np.pi, m.ndim / 2) * np.sqrt(np.linalg.det(m.cov))

    def marginal(self, dim):
        """Return the marginal for a given dimension."""
        return Gaussian(self.mean[np.ix_(dim, [0])],
                        self.cov[np.ix_(dim, dim)])

    def log(self):
        """Return the natural logarithm of the random variable."""
        # TODO: Not implemented for Gaussian random variable.
        raise NotImplementedError
