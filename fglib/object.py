"""Module for factor graph objects.

This module contains node classes and edge classes,
which are used to build factor graphs.

Classes:
Node -- Abstract node
VNode -- Variable node
IOVNode -- Input-output variable node
FNode -- Factor node
IOFNode -- Input-output factor node
Edge -- Edge

"""

import types

import networkx as nx


class Node(object):

    """Abstract base class for all nodes."""

    def __init__(self, label):
        """Create a node with an associated label."""
        self.label = str(label)

    def __str__(self):
        """Return string representation."""
        return self.label

    def neighbors(self, graph, node, exclusion=None):
        """Get all neighbors with a given exclusion.

        Return iterator over all neighboring nodes
        without the given exclusion node.

        Positional arguments:
        graph -- the factor graph containing the node
        node -- the node
        exclusion -- the exclusion node

        """

        if exclusion is None:
            return nx.all_neighbors(graph, node)
        else:
            # Build iterator set
            iterator = (exclusion,) \
                if not isinstance(exclusion, list) else exclusion

            # Return neighbors excluding iterator set
            return (n for n in nx.all_neighbors(graph, node)
                    if n not in iterator)


class VNode(Node):

    """Variable node.

    Variable node inherited from node base class.
    Extends the base class with message passing methods.

    """

    def __init__(self, label, init=None, observed=False):
        """Create a variable node."""
        Node.__init__(self, label)
        self.TYPE = "vn"
        self.init = init
        self.observed = observed

    def belief(self, graph):
        """Return belief of the variable node."""
        iterator = graph.neighbors_iter(self)

        # Pick first node
        n = next(iterator)

        # Product over all incoming messages
        belief = graph[n][self]['object'].get_message(n, self)
        for n in iterator:
            belief *= graph[n][self]['object'].get_message(n, self)

        return belief

    def spa(self, graph, tnode):
        """Return message of the sum-product algorithm."""
        if self.observed:
            return self.init
        else:
            # Initial message
            msg = self.init

            # Product over incoming messages
            for n in self.neighbors(graph, self, tnode):
                msg *= graph[n][self]['object'].get_message(n, self)

            return msg

    def mpa(self, graph, tnode):
        """Return message of the max-product algorithm."""
        return self.spa(graph, tnode)

    def msa(self, graph, tnode):
        """Return message of the max-sum algorithm."""
        if self.observed:
            return self.init
        else:
            # Initial message
            msg = self.init
            msg = msg.log()

            # Sum over incoming messages
            for n in self.neighbors(graph, self, tnode):
                msg += graph[n][self]['object'].get_message(n, self)

            return msg

    def mf(self, graph, tnode):
        """Return message of the mean-field algorithm."""
        if self.observed:
            return self.init
        else:
            return self.belief(graph)

    def marginal(self, graph, norm=True):
        """Return the marginal distribution of the variable node."""
        b = self.belief(graph)

        if norm:  # return normalized marginal
            if graph.norm_const is None:  # compute normalization constant
                graph.norm_const = 1 / sum(b.value)
            return graph.norm_const * b.value
        return b

    def maximum(self, graph):
        """Return the maximum probability of the variable node."""
        b = self.belief(graph)
        return b.max(self)

    def argmax(self, graph):
        """Return the argument for maximum probability of the variable node."""
        # In case of multiple occurrences of the maximum values,
        # the indices corresponding to the first occurrence are returned.
        return self.belief(graph).argmax(self)


class IOVNode(VNode):

    """Input-output variable node.

    Input-output variable node inherited from variable node class.
    Overwrites all message passing methods of the base class
    with a given callback function.

    """

    def __init__(self, label, init=None, observed=False, callback=None):
        """Create an input-output variable node."""
        VNode.__init__(self, label, init, observed)
        if callback is not None:
            self.set_callback(callback)

    def set_callback(self, callback):
        """Set callback function.

        Add bounded methods to the class instance in order to overwrite
        the existing message passing methods.

        """
        self.spa = types.MethodType(callback, self)
        self.mpa = types.MethodType(callback, self)
        self.msa = types.MethodType(callback, self)
        self.mf = types.MethodType(callback, self)


class FNode(Node):

    """Factor node.

    Factor node inherited from node base class.
    Extends the base class with message passing methods.

    """

    def __init__(self, label, factor):
        """Create a factor node."""
        Node.__init__(self, label)
        self.TYPE = "fn"
        self.factor = factor
        self.record = {}

    def spa(self, graph, tnode):
        """Return message of the sum-product algorithm."""
        # Initialize with local factor
        msg = self.factor

        # Product over incoming messages
        for n in self.neighbors(graph, self, tnode):
            msg *= graph[n][self]['object'].get_message(n, self)

        # Integration/Summation over incoming variables
        for n in self.neighbors(graph, self, tnode):
            msg = msg.int(n)

        return msg

    def mpa(self, graph, tnode):
        """Return message of the max-product algorithm."""
        self.record[tnode] = {}

        # Initialize with local factor
        msg = self.factor

        # Product over incoming messages
        for n in self.neighbors(graph, self, tnode):
            msg *= graph[n][self]['object'].get_message(n, self)

        # Maximization over incoming variables
        for n in self.neighbors(graph, self, tnode):
            self.record[tnode][n] = msg.argmax(n)  # Record for back-tracking
            msg = msg.max(n)

        return msg

    def msa(self, graph, tnode):
        """Return message of the max-sum algorithm."""
        self.record[tnode] = {}

        # Initialize with logarithm of local factor
        msg = self.factor.log()

        # Sum over incoming messages
        for n in self.neighbors(graph, self, tnode):
            msg += graph[n][self]['object'].get_message(n, self)

        # Maximization over incoming variables
        for n in self.neighbors(graph, self, tnode):
            self.record[tnode][n] = msg.argmax(n)  # Record for back-tracking
            msg = msg.max(n)

        return msg

    def mf(self, graph, tnode):
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

    def __init__(self, label, factor, callback=None):
        """Create an input-output factor node."""
        FNode.__init__(self, label, factor)
        if callback is not None:
            self.set_callback(callback)

    def set_callback(self, callback):
        """Set callback function.

        Add bounded methods to the class instance in order to overwrite
        the existing message passing methods.

        """
        self.spa = types.MethodType(callback, self)
        self.mpa = types.MethodType(callback, self)
        self.msa = types.MethodType(callback, self)
        self.mf = types.MethodType(callback, self)


class Edge(object):

    """Edge.

    Base class for all edges.
    Each edge class contains a message attribute,
    which stores the corresponding message in forward and backward direction.

    """

    def __init__(self, snode, tnode, init=None):
        """Create an edge."""
        # Array Index
        self.index = {snode: 0, tnode: 1}

        # Two-dimensional message list
        self.message = [[None, init], \
                        [init, None]]

        # Variable node
        if snode.TYPE == "vn":
            self.variable = snode
        else:
            self.variable = tnode

    def __str__(self):
        """Return string representation."""
        return str(self.message)

    def set_message(self, snode, tnode, value):
        """Set value of message from source node to target node."""
        self.message[self.index[snode]][self.index[tnode]] = value

    def get_message(self, snode, tnode):
        """Return value of message from source node to target node."""
        return self.message[self.index[snode]][self.index[tnode]]
