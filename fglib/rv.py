"""Module for random variables.

This module contains classes for random variables and exceptions.

Classes:
    ParameterException: Exception for invalid parameters.
    Discrete: Class for discrete random variables.
    Gaussian: Class for Gaussian random variables.

"""

import numpy as np


class ParameterException(Exception):

    """Exception for invalid parameters."""

    pass


class Discrete(object):

    """Class for discrete random variables.

    A discrete random variable is defined by a single- or multi-dimensional
    probability mass function. In addition, each dimension of the probability
    mass function has to be associated with a variable. The variable is
    represented by a variable node of the comprehensive factor graph.

    """

    def __init__(self, raw_pmf, *args):
        """Initialize a discrete random variable.

        Create a new discrete random variable with the given probability
        mass function over the given variable nodes.

        Args:
            raw_pmf: A Numpy array representing the probability mass function.
            *args: Instances of the class VNode representing the variables of
                the probability mass function. The number of the positional
                arguments must match the number of dimensions of the Numpy
                array.

        Raises:
            ParameterException: An error occurred initializing with invalid
                parameters.

        """
        pmf = np.asarray(raw_pmf, dtype=np.float64)

        # Set probability mass function
        if not np.allclose(np.sum(pmf), 1):
            raise ParameterException('Invalid probability mass function.')
        else:
            self._pmf = pmf

        # Set variable nodes for dimensions
        if np.ndim(pmf) != len(args):
            raise ParameterException('Dimension mismatch.')
        else:
            self._dim = args

    @property
    def pmf(self):
        return self._pmf

    @property
    def dim(self):
        return self._dim

    def __str__(self):
        """Return string representation of the discrete random variable."""
        return str(self.pmf)

    def __add__(self, other):
        """Add other to self and return the result.

        Args:
            other: Summand for the discrete random variable.

        Returns:
            A new discrete random variable representing the summation.

        """
        pmf = np.convolve(self.pmf, other.pmf, 'same')

        return Discrete(pmf, self.dim)

    def __sub__(self, other):
        """Subtract other from self and return the result.

        Args:
            other: Subtrahend for the discrete random variable.

        Returns:
            A new discrete random variable representing the summation

        """
        pmf = np.convolve(self.pmf[::-1], other.pmf, 'same')

        return Discrete(pmf, self.dim)

    def __mul__(self, other):
        """Multiply other with self and return the result.

        Args:
            other: Multiplier for the discrete random variable.

        Returns:
            A new discrete random variable representing the multiplication.

        """
        # Verify dimensions of multiplicand and multiplier.
        if len(self.dim) < len(other.dim):
            self._expand(other.dim)
        elif len(self.dim) > len(other.dim):
            other._expand(self.dim)

        # Normalize probability mass function.
        pmf = self.pmf * other.pmf
        pmf /= np.sum(pmf)

        return Discrete(pmf, *self.dim)

    def __iadd__(self, other):
        """Method for augmented addition.

        Args:
            other: Summand for the discrete random variable.

        Returns:
            A new discrete random variable representing the summation.

        """
        return self.__add__(other)

    def __isub__(self, other):
        """Method for augmented subtraction.

        Args:
            other: Subtrahend for the discrete random variable.

        Returns:
            A new discrete random variable representing the summation

        """
        return self.__sub__(other)

    def __imul__(self, other):
        """Method for augmented multiplication.

        Args:
            other: Multiplier for the discrete random variable.

        Returns:
            A new discrete random variable representing the multiplication.

        """
        return self.__mul__(other)

    def __eq__(self, other):
        """Compare self with other and return the boolean result.

        Two discrete random variables are equal only if the probability mass
        functions are equal and the order of dimensions are equal.

        """
        return np.allclose(self.pmf, other.pmf) \
            and self.dim == other.dim

    def _expand(self, dims):
        """Expand dimensions.

        Expand the discrete random variable along the given new dimensions.

        Args:
            dims: List of discrete random variables.

        """
        reps = [1, ] * len(dims)

        # Extract missing dimensions
        diff = [i for i, d in enumerate(dims) if d not in self.dim]

        # Expand missing dimensions
        for d in diff:
            self._pmf = np.expand_dims(self.pmf, axis=d)
            reps[d] = 2

        # Repeat missing dimensions
        self._pmf = np.tile(self.pmf, reps)
        self._dim = dims

    def marginalize(self, *dims):
        """Return the marginal for given dimensions.

        The probability mass function of the discrete random variable is
        marginalized along the given dimensions.

        Args:
            *dims: Instances of discrete random variables, which should be
                marginalized out.

        Returns:
            A new discrete random variable representing the marginal.

        """
        axis = tuple(idx for idx, d in enumerate(self.dim) if d in dims)
        pmf = np.sum(self.pmf, axis)
        pmf /= np.sum(pmf)

        new_dims = tuple(d for d in self.dim if d not in dims)
        return Discrete(pmf, *new_dims)

    def maximize(self, *dims):
        """Return the maximum for given dimensions.

        The probability mass function of the discrete random variable is
        maximized along the given dimensions.

        Args:
            *dims: Instances of discrete random variables, which should be
                maximized out.

        Returns:
            A new discrete random variable representing the maximum.

        """
        axis = tuple(idx for idx, d in enumerate(self.dim) if d in dims)
        pmf = np.amax(self.pmf, axis)
        pmf /= np.sum(pmf)

        new_dims = tuple(d for d in self.dim if d not in dims)
        return Discrete(pmf, *new_dims)

    def argmax(self, dim=None):
        """Return the dimension index of the maximum.

        Args:
            dim: An optional discrete random variable along a marginalization
                should be performed and the maximum is searched over the
                remaining dimensions. In the case of None, the maximum is
                search along all dimensions.

        Returns:
            An integer representing the dimension of the maximum.

        """
        if dim is None:
            return np.unravel_index(self.pmf.argmax(), self.pmf.shape)
        m = self.marginalize(dim)
        return np.argmax(m.pmf)

    def log(self):
        """Natural logarithm of the discrete random variable.

        Returns:
            A new discrete random variable with the natural logarithm of the
            probablitiy mass function.

        """
        return Discrete(np.log(self.value), self.dim)


class Gaussian(object):

    """Class for Gaussian random variables.

    A Gaussian random variable is defined by a mean vector and a covariance
    matrix. In addition, each dimension of the mean vector and the covariance
    matrix has to be associated with a variable. The variable is
    represented by a variable node of the comprehensive factor graph.

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
        m = self.marginalize(dim)
        return np.power(2 * np.pi, m.ndim / 2) * np.sqrt(np.linalg.det(m.cov))

    def marginalize(self, dim):
        """Return the marginalize for a given dimension."""
        return Gaussian(self.mean[np.ix_(dim, [0])],
                        self.cov[np.ix_(dim, dim)])

    def log(self):
        """Return the natural logarithm of the random variable."""
        # TODO: Not implemented for Gaussian random variable.
        raise NotImplementedError
