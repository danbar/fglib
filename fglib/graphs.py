"""Module for factor graphs.

This module contains the base class for factor graphs
with additional functions to convert arbitrary graphs to factor graphs.

Classes:
    FactorGraph: Class for factor graphs.

Functions:
    convert_graph_to_factor_graph: Convert bipartite graph to factor graph.

"""

import networkx as nx

from . import nodes


class FactorGraph(nx.DiGraph):

    """Class for factor graphs.

    A factor graph represents the factorization of a function of
    several variables. Assume, for example, that some function
    f(x1, x2, x3, x4) can be factored as

        f(x1, x2, x3, x4) = fa(x1, x2) fb(x2, x3) fc(x2, x4).

    The factor graph representing the factorization is given as

     //--\\     +----+     //--\\     +----+     //--\\
     | x1 |-----| fa |-----| x2 |-----| fb |-----| x3 |
     \\--//     +----+     \\--//     +----+     \\--//
                             |
                           +----+
                           | fc |
                           +----+
                             |
                           //--\\
                           | x4 |
                           \\--//

    The class for factor graphs is inherited from the base class
    for directed graphs (of the NetworkX library).

    """

    def __init__(self):
        """Initialize a factor graph."""
        super().__init__(self, name="Factor Graph")

    def set_node(self, node):
        """Set a single node to the factor graph.

        Args:
            node: A single node

        """
        self.add_node(node)

    def set_nodes(self, nodes):
        """Set multiple nodes to the factor graph.

        Args:
            nodes: A list of multiple nodes

        """
        for n in nodes:
            self.set_node(n)

    def get_nodes(self):
        """ Get multiple nodes from the factor graph.

        Returns:
            A list of all nodes.

        """
        return [n for n in self.nodes()]

    def get_vnodes(self):
        """Get all variable nodes of the factor graph.

        Returns:
            A list of all variable nodes.

        """
        return [n for n in self.nodes()
                if n.type == nodes.NodeType.variable_node]

    def get_fnodes(self):
        """Get all factor nodes of the factor graph.

        Returns:
            A list of all factor nodes.

        """
        return [n for n in self.nodes()
                if n.type == nodes.NodeType.factor_node]

    def set_edge(self, snode, tnode, init=None):
        """Set a single edge to the factor graph.

        A single edge is added to the factor graph.
        It can be initialized with a given random variable.

        Args:
            snode: Source node for edge
            tnode: Target node for edge
            init: Initial message for edge

        """
        self.add_edge(snode, tnode, msg=init)
        self.add_edge(tnode, snode, msg=init)

    def set_edges(self, edges):
        """Set multiple edges to the factor graph.

        Args:
            edges: A list of multiple edges

        """
        for (snode, tnode) in edges:
            self.set_edge(snode, tnode)

    def get_message(self, snode, tnode):
        """Get a single edge from the factor graph.

        Args:
            slabel: Source label for edge
            tlabel: Target label for edge

        Returns:
            A single edge.

        """
        return self.edges[snode, tnode]['msg']

    def get_incoming_messages(self, node, exclude_node=None):
        """Get multiple edges from the factor graph:

        Returns:
            A list of multiple edges.

        """
        if exclude_node is None:
            return [d['msg'] for (_, _, d) in self.in_edges(node, data=True)]
        else:
            return [d['msg'] for (u, v, d) in self.in_edges(node, data=True)
                    if exclude_node is not u and exclude_node is not v]


class ForneyFactorGraph(FactorGraph):

    """Class  for Forney-style factor graphs.

    A Forney-style factor graph represents the factorization of a function of
    several variables. Assume, for example, that some function
    f(x1, x2, x3, x4) can be factored as

        f(x1, x2, x3, x4) = fa(x1, x2) fb(x2, x3) fc(x2, x4).

    The factor graph representing the factorization is given as

      x1  +----+  x2 +----+ x2' +----+  x3
     -----| fa |-----| =  |-----| fb |-----
          +----+     +----+     +----+
                       |
                       | x2''
                       |
                     +----+
                     | fc |
                     +----+
                       |
                       | x4.
                       |

    The class for Forney-style factor graphs is inherited from the base class
    for factor graphs.

    """

    def __init__(self):
        """Initialize a Forney-style factor graph."""
        super().__init__(self)

    # TODO: Needs to be implemented!


def convert_graph_to_factor_graph(graph, vnode, fnode, rv_type):
    """Convert bipartite graph to factor graph.

    Convert a bipartite graph from the NetworkX library to a factor graph.
    For the bipartite graph, all nodes with label 'bipartite' equal to 0 are
    replaced by instances of the given variable node class and all nodes with
    label 'bipartite' equal to 1 are replaced by instances of the given factor
    node class.

    Args:
        graph: Bipartite graph used for conversion.
        vnode: Variable node class.
        fnode: Factor node class.
        rv_type: Type of random variable for variable nodes.

    Returns:
        A factor graph.

    """
    # Initialize factor graph
    fgraph = FactorGraph()

    # Create mapping
    mapping = dict(zip(graph, graph))

    # Insert variable nodes into mapping
    vn = [n for (n, d) in graph.nodes(data=True) if d['bipartite'] == 0]
    vn_instances = [vnode(label, rv_type) for _, label in enumerate(vn)]
    mapping.update((n, vn_instances.pop()) for n in vn)

    # Insert factor nodes into mapping
    fn = [n for (n, d) in graph.nodes(data=True) if d['bipartite'] == 1]
    fn_instances = [fnode(label) for _, label in enumerate(fn)]
    mapping.update((n, fn_instances.pop()) for n in fn)

    # Map graph to factor graph
    graph = nx.relabel_nodes(graph, mapping)  # Returns a copy
    fgraph.set_nodes(graph.nodes())
    fgraph.set_edges(graph.edges())

    return fgraph
