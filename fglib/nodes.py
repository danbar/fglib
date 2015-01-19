"""Module for factor graph nodes.

This module contains node classes,
which are used to build factor graphs.

Classes:
Node -- Abstract node
VNode -- Variable node
IOVNode -- Input-output variable node
FNode -- Factor node
IOFNode -- Input-output factor node

"""

from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum
from types import MethodType

import networkx as nx


class NodeType(Enum):

    variable_node = 1
    factor_node = 2


class Node(metaclass=ABCMeta):

    """Abstract base class for all nodes."""

    def __init__(self, label, graph=None):
        """Create a node with an associated label."""
        self.__label = str(label)
        self.graph = graph

    def __str__(self):
        """Return string representation."""
        return self.__label

    @abstractproperty
    def type(self):
        pass

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph):
        self.__graph = graph

    def neighbors(self, exclusion=None):
        """Get all neighbors with a given exclusion.

        Return iterator over all neighboring nodes
        without the given exclusion node.

        Positional arguments:
        exclusion -- the exclusion node

        """

        if exclusion is None:
            return nx.all_neighbors(self.graph, self)
        else:
            # Build iterator set
            iterator = (exclusion,) \
                if not isinstance(exclusion, list) else exclusion

            # Return neighbors excluding iterator set
            return (n for n in nx.all_neighbors(self.graph, self)
                    if n not in iterator)

    @abstractmethod
    def spa(self, tnode):
        pass

    @abstractmethod
    def mpa(self, tnode):
        pass

    @abstractmethod
    def msa(self, tnode):
        pass

    @abstractmethod
    def mf(self, tnode):
        pass


class VNode(Node):

    """Variable node.

    Variable node inherited from node base class.
    Extends the base class with message passing methods.

    """

    def __init__(self, label, graph=None, init=None, observed=False):
        """Create a variable node."""
        super().__init__(label, graph)
        self.init = init
        self.observed = observed

    @property
    def type(self):
        return NodeType.variable_node

    def belief(self):
        """Return belief of the variable node."""
        iterator = self.graph.neighbors_iter(self)

        # Pick first node
        n = next(iterator)

        # Product over all incoming messages
        belief = self.graph[n][self]['object'].get_message(n, self)
        for n in iterator:
            belief *= self.graph[n][self]['object'].get_message(n, self)

        return belief

    def marginal(self, norm=True):
        """Return the marginal distribution of the variable node."""
        b = self.belief(self.graph)

        if norm:  # return normalized marginal
            if self.graph.norm_const is None:  # compute normalization constant
                self.graph.norm_const = 1 / sum(b.value)
            return self.graph.norm_const * b.value
        return b

    def maximum(self):
        """Return the maximum probability of the variable node."""
        b = self.belief(self.graph)
        return b.max(self)

    def argmax(self):
        """Return the argument for maximum probability of the variable node."""
        # In case of multiple occurrences of the maximum values,
        # the indices corresponding to the first occurrence are returned.
        return self.belief(self.graph).argmax(self)

    def spa(self, tnode):
        """Return message of the sum-product algorithm."""
        if self.observed:
            return self.init
        else:
            # Initial message
            msg = self.init

            # Product over incoming messages
            for n in self.neighbors(self.graph, self, tnode):
                msg *= self.graph[n][self]['object'].get_message(n, self)

            return msg

    def mpa(self, tnode):
        """Return message of the max-product algorithm."""
        return self.spa(self.graph, tnode)

    def msa(self, tnode):
        """Return message of the max-sum algorithm."""
        if self.observed:
            return self.init
        else:
            # Initial message
            msg = self.init
            msg = msg.log()

            # Sum over incoming messages
            for n in self.neighbors(self.graph, self, tnode):
                msg += self.graph[n][self]['object'].get_message(n, self)

            return msg

    def mf(self, tnode):
        """Return message of the mean-field algorithm."""
        if self.observed:
            return self.init
        else:
            return self.belief(self.graph)


class IOVNode(VNode):

    """Input-output variable node.

    Input-output variable node inherited from variable node class.
    Overwrites all message passing methods of the base class
    with a given callback function.

    """

    def __init__(self, label, graph, init=None, observed=False, callback=None):
        """Create an input-output variable node."""
        super().__init__(label, graph, init, observed)
        if callback is not None:
            self.set_callback(callback)

    def set_callback(self, callback):
        """Set callback function.

        Add bounded methods to the class instance in order to overwrite
        the existing message passing methods.

        """
        self.spa = MethodType(callback, self)
        self.mpa = MethodType(callback, self)
        self.msa = MethodType(callback, self)
        self.mf = MethodType(callback, self)


class FNode(Node):

    """Factor node.

    Factor node inherited from node base class.
    Extends the base class with message passing methods.

    """

    def __init__(self, label, factor, graph=None):
        """Create a factor node."""
        super().__init__(label, graph)
        self.factor = factor
        self.record = {}

    @property
    def type(self):
        return NodeType.factor_node

    def spa(self, tnode):
        """Return message of the sum-product algorithm."""
        # Initialize with local factor
        msg = self.factor

        # Product over incoming messages
        for n in self.neighbors(self.graph, self, tnode):
            msg *= self.graph[n][self]['object'].get_message(n, self)

        # Integration/Summation over incoming variables
        for n in self.neighbors(self.graph, self, tnode):
            msg = msg.int(n)

        return msg

    def mpa(self, tnode):
        """Return message of the max-product algorithm."""
        self.record[tnode] = {}

        # Initialize with local factor
        msg = self.factor

        # Product over incoming messages
        for n in self.neighbors(self.graph, self, tnode):
            msg *= self.graph[n][self]['object'].get_message(n, self)

        # Maximization over incoming variables
        for n in self.neighbors(self.graph, self, tnode):
            self.record[tnode][n] = msg.argmax(n)  # Record for back-tracking
            msg = msg.max(n)

        return msg

    def msa(self, tnode):
        """Return message of the max-sum algorithm."""
        self.record[tnode] = {}

        # Initialize with logarithm of local factor
        msg = self.factor.log()

        # Sum over incoming messages
        for n in self.neighbors(self.graph, self, tnode):
            msg += self.graph[n][self]['object'].get_message(n, self)

        # Maximization over incoming variables
        for n in self.neighbors(self.graph, self, tnode):
            self.record[tnode][n] = msg.argmax(n)  # Record for back-tracking
            msg = msg.max(n)

        return msg

    def mf(self, tnode):
        """Return message of the mean-field algorithm."""
        # Initialize with local factor
        msg = self.factor

#TODO:    # Product over incoming messages
#         for n in self.neighbors(graph, self, tnode):
#             msg *= graph[n][self]['object'].get_message(n, self)
#
#         # Integration/Summation over incoming variables
#         for n in self.neighbors(graph, self, tnode):
#             msg = msg.int(n)

        return msg


class IOFNode(FNode):

    """Input-output factor node.

    Input-output factor node inherited from factor node class.
    Overwrites all message passing methods of the base class
    with a given callback function.

    """

    def __init__(self, label, factor, graph=None, callback=None):
        """Create an input-output factor node."""
        super().__init__(self, label, factor, graph)
        if callback is not None:
            self.set_callback(callback)

    def set_callback(self, callback):
        """Set callback function.

        Add bounded methods to the class instance in order to overwrite
        the existing message passing methods.

        """
        self.spa = MethodType(callback, self)
        self.mpa = MethodType(callback, self)
        self.msa = MethodType(callback, self)
        self.mf = MethodType(callback, self)
