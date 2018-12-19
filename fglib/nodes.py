"""Module for nodes of factor graphs.

This module contains classes for nodes of factor graphs,
which are used to build factor graphs.

Classes:
    Node: Abstract class for nodes.
    VNode: Class for variable nodes.
    IOVNode: Class for custom input-output variable nodes.
    FNode: Class for factor nodes.
    IOFNode: Class for custom input-output factor nodes.

"""

from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from types import MethodType


class NodeType(Enum):

    """Enumeration for node types."""

    variable_node = 1
    factor_node = 2


class Node(ABC):

    """Abstract base class for all nodes."""

    def __init__(self, label):
        """Create a node with an associated label."""
        self.__label = str(label)

    def __str__(self):
        """Return string representation."""
        return self.__label

    @abstractproperty
    def type(self):
        """Specify the NodeType."""

    @property
    def label(self):
        return self.__label

    @abstractmethod
    def spa(self, tnode, msgs_in):
        """Return message of the sum-product algorithm."""

    @abstractmethod
    def mpa(self, tnode, msgs_in):
        """Return message of the max-product algorithm."""

    @abstractmethod
    def msa(self, tnode, msgs_in):
        """Return message of the max-sum algorithm."""

    @abstractmethod
    def mf(self, tnode, msgs_in):
        """Return message of the mean-field algorithm."""


class VNode(Node):

    """Variable node.

    Variable node inherited from node base class.
    Extends the base class with message passing methods.

    """

    def __init__(self, label, rv_type, observed=False):
        """Create a variable node."""
        super().__init__(label)
        self.init = rv_type.unity(self)
        self.observed = observed

    @property
    def type(self):
        return NodeType.variable_node

    @property
    def init(self):
        return self.__init

    @init.setter
    def init(self, init):
        self.__init = init

    def belief(self, msgs_in, normalize=True, logarithmic=False):
        """Return belief of the variable node.

        Args:
            normalize: Boolean flag if belief should be normalized.

        """
        # Pick first message
        it_msgs = iter(msgs_in)
        belief = next(it_msgs)

        # Product over all incoming messages
        if not logarithmic:
            for msg_in in it_msgs:
                belief *= msg_in
        else:
            for msg_in in it_msgs:
                belief += msg_in

        if normalize:
            belief = belief.normalize()

        return belief

    def max(self, msgs_in, normalize=True, logarithmic=False):
        """Return the maximum probability of the variable node.

        Args:
            normalize: Boolean flag if belief should be normalized.

        """
        b = self.belief(msgs_in, normalize, logarithmic)
        return b.max()

    def argmax(self, msgs_in):
        """Return the argument for maximum probability of the variable node."""
        # In case of multiple occurrences of the maximum values,
        # the indices corresponding to the first occurrence are returned.
        b = self.belief(msgs_in)
        return b.argmax()

    def spa(self, tnode, msgs_in):
        """Return message of the sum-product algorithm."""
        if self.observed:
            return self.init
        else:
            # Initial message
            msg_out = self.init

            # Product over incoming messages
            for msg_in in msgs_in:
                msg_out *= msg_in

            return msg_out

    def mpa(self, tnode, msgs_in):
        """Return message of the max-product algorithm."""
        return self.spa(tnode, msgs_in)

    def msa(self, tnode, msgs_in):
        """Return message of the max-sum algorithm."""
        if self.observed:
            return self.init.log()
        else:
            # Initial (logarithmized) message
            msg_out = self.init.log()

            # Sum over incoming messages
            for msg_in in msgs_in:
                msg_out += msg_in

            return msg_out

    def mf(self, tnode, msgs_in):
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

    def __init__(self, label, init=None, observed=False, callback=None):
        """Create an input-output variable node."""
        super().__init__(label, init, observed)
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

    def __init__(self, label, factor=None):
        """Create a factor node."""
        super().__init__(label)
        self.factor = factor
        self.record = {}

    @property
    def type(self):
        return NodeType.factor_node

    @property
    def factor(self):
        return self.__factor

    @factor.setter
    def factor(self, factor):
        self.__factor = factor

    def spa(self, tnode, msgs_in):
        """Return message of the sum-product algorithm."""
        # Initialize with local factor
        msg_out = self.factor

        # Product over incoming messages
        for msg_in in msgs_in:
            msg_out *= msg_in

        # Integration/Summation over incoming variables
        for msg_in in msgs_in:
            msg_out = msg_out.marginalize(msg_in, normalize=False)

        return msg_out

    def mpa(self, tnode, msgs_in):
        """Return message of the max-product algorithm."""
        self.record[tnode] = {}  # TODO: Can we avoid 'tnode'?

        # Initialize with local factor
        msg_out = self.factor

        # Product over incoming messages
        for msg_in in msgs_in:
            msg_out *= msg_in

        # Maximization over incoming variables
        for msg_in in msgs_in:
            self.record[tnode][msg_in.dim] = msg_out.argmax()  # Back-tracking
            msg_out = msg_out.maximize(msg_in, normalize=False)

        return msg_out

    def msa(self, tnode, msgs_in):
        """Return message of the max-sum algorithm."""
        self.record[tnode] = {}

        # Initialize with (logarithmized) local factor
        msg_out = self.factor.log()

        # Sum over incoming messages
        for msg_in in msgs_in:
            msg_out += msg_in

        # Maximization over incoming variables
        for msg_in in msgs_in:
            self.record[tnode][msg_in.dim] = msg_out.argmax()  # Back-tracking
            msg_out = msg_out.maximize(msg_in, normalize=False)

        return msg_out

    def mf(self, tnode, msgs_in):
        """Return message of the mean-field algorithm."""
        # Initialize with local factor
        msg = self.factor

#         # Product over incoming messages
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

    def __init__(self, label, factor, callback=None):
        """Create an input-output factor node."""
        super().__init__(self, label, factor)
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
