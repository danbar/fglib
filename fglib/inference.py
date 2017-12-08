"""Module for inference algorithms.

This module contains different functions to perform inference on factor graphs.

Functions:
    belief_propagation: Belief propagation
    sum_product: Sum-product algorithm
    max_product: Max-product algorithm
    max_sum: Max-sum algorithm
    loopy_belief_propagation: Loopy belief propagation
    mean_field: Mean-field algorithm

"""

from random import choice

import networkx as nx

from . import nodes


def belief_propagation(graph, query_node=None):
    """Belief propagation.

    Perform exact inference on tree structured graphs.
    Return the belief of all query_nodes.

    """

    if query_node is None:  # pick random node
        query_node = choice(graph.get_vnodes())

    # Depth First Search to determine edges
    dfs = nx.dfs_edges(graph, query_node)

    # Convert tuple to reversed list
    backward_path = list(dfs)
    forward_path = reversed(backward_path)

    # Messages in forward phase
    for (v, u) in forward_path:  # Edge direction: u -> v
        msg = u.spa(v)
        graph[u][v]['object'].set_message(u, v, msg)

    # Messages in backward phase
    for (u, v) in backward_path:  # Edge direction: u -> v
        msg = u.spa(v)
        graph[u][v]['object'].set_message(u, v, msg)

    # Return marginal distribution
    return query_node.belief()


def sum_product(graph, query_node=None):
    """Sum-product algorithm.

    Compute marginal distribution on graphs that are tree structured.
    Return the belief of all query_nodes.

    """

    # Sum-Product algorithm is equivalent to Belief Propagation
    return belief_propagation(graph, query_node)


def max_product(graph, query_node=None):
    """Max-product algorithm.

    Compute setting of variables with maximum probability on graphs
    that are tree structured.
    Return the setting of all query_nodes.

    """
    track = {}  # Setting of variables

    if query_node is None:  # pick random node
        query_node = choice(graph.get_vnodes())

    # Depth First Search to determine edges
    dfs = nx.dfs_edges(graph, query_node)

    # Convert tuple to reversed list
    backward_path = list(dfs)
    forward_path = reversed(backward_path)

    # Messages in forward phase
    for (v, u) in forward_path:  # Edge direction: u -> v
        msg = u.mpa(v)
        graph[u][v]['object'].set_message(u, v, msg)

    # Messages in backward phase
    for (u, v) in backward_path:  # Edge direction: u -> v
        msg = u.mpa(v)
        graph[u][v]['object'].set_message(u, v, msg)

    # Maximum argument for query node
    track[query_node] = query_node.argmax()

    # Back-tracking
    for (u, v) in backward_path:  # Edge direction: u -> v
        if v.type == nodes.NodeType.factor_node:
            for k in v.record[u].keys():  # Iterate over outgoing edges
                track[k] = v.record[u][k]

    # Return maximum probability for query node and setting of variable
    return query_node.maximum(), track


def max_sum(graph, query_node=None):
    """Max-sum algorithm.

    Compute setting of variable for maximum probability on graphs
    that are tree structured.
    Return the setting of all query_nodes.

    """
    track = {}  # Setting of variables

    if query_node is None:  # pick random node
        query_node = choice(graph.get_vnodes())

    # Depth First Search to determine edges
    dfs = nx.dfs_edges(graph, query_node)

    # Convert tuple to reversed list
    backward_path = list(dfs)
    forward_path = reversed(backward_path)

    # Messages in forward phase
    for (v, u) in forward_path:  # Edge direction: u -> v
        msg = u.msa(v)
        graph[u][v]['object'].set_message(u, v, msg, logarithmic=True)

    # Messages in backward phase
    for (u, v) in backward_path:  # Edge direction: u -> v
        msg = u.msa(v)
        graph[u][v]['object'].set_message(u, v, msg, logarithmic=True)

    # Maximum argument for query node
    track[query_node] = query_node.argmax()

    # Back-tracking
    for (u, v) in backward_path:  # Edge direction: u -> v
        if v.type == nodes.NodeType.factor_node:
            for k in v.record[u].keys():  # Iterate over outgoing edges
                track[k] = v.record[u][k]

    # Return maximum probability for query node and setting of variable
    return query_node.maximum(), track


def loopy_belief_propagation(model, iterations, query_node=(), order=None):
    """Loopy belief propagation.

    Perform approximative inference on arbitrary structured graphs.
    Return the belief of all query_nodes.

    """
    if order is None:
        fn = [n for (n, attr) in model.nodes(data=True)
              if attr["type"] == "fn"]
        vn = [n for (n, attr) in model.nodes(data=True)
              if attr["type"] == "vn"]
        order = fn + vn
    return _schedule(model, 'spa', iterations, query_node, order)


def mean_field(model, iterations, query_node=(), order=None):
    """Mean-field algorithm.

    Perform approximative inference on arbitrary structured graphs.
    Return the belief of all query_nodes.

    """
    if order is None:
        fn = [n for (n, attr) in model.nodes(data=True)
              if attr["type"] == "fn"]
        vn = [n for (n, attr) in model.nodes(data=True)
              if attr["type"] == "vn"]
        order = fn + vn
    return _schedule(model, 'mf', iterations, query_node, order)


def _schedule(model, method, iterations, query_node, order):
    """Flooding schedule.

    A flooding scheduler for factor graphs with cycles.
    A given number of iterations is performed in a defined node order.
    Return the belief of all query_nodes.

    """
    b = {n: [] for n in query_node}

    # Iterative message passing
    for _ in range(iterations):

        # Visit nodes in predefined order
        for n in order:
            for neighbor in nx.all_neighbors(model, n):
                msg = getattr(n, method)(model, neighbor)
                model[n][neighbor]['object'].set_message(n, neighbor, msg)

        # Beliefs of query nodes
        for n in query_node:
            b[n].append(n.belief(model))

    return b
