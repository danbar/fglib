"""Module for edges of factor graphs.

This module contains classes for edges of factor graphs,
which are used to build factor graphs.

Classes:
    Edge: Class for edges.

"""

from . import nodes


class Edge:

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
        self.message = [[None, init],
                        [init, None]]

        self.logarithmic = False

        # Variable node
        if snode.type == nodes.NodeType.variable_node:
            self.variable = snode
        else:
            self.variable = tnode

    def __str__(self):
        """Return string representation."""
        return str(self.message)

    def set_message(self, snode, tnode, value, logarithmic=False):
        """Set value of message from source node to target node."""
        self.message[self.index[snode]][self.index[tnode]] = value
        self.logarithmic = logarithmic

    def get_message(self, snode, tnode):
        """Return value of message from source node to target node."""
        return self.message[self.index[snode]][self.index[tnode]]
