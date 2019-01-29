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


def _message_passing(graph, query_node, algorithm):
    """Message passing.

    ...

    """

    # Depth First Search to determine edges
    # and convert tuple to a (reversed) list
    dfs = nx.dfs_edges(graph, query_node)

    backward_path = list(dfs)
    forward_path = reversed(backward_path)

    # Messages in forward phase
    for (v, u) in forward_path:  # Edge direction: u -> v
        msgs_in = graph.get_incoming_messages(u, exclude_node=v)
        msg_out = getattr(u, algorithm)(v, msgs_in)
        graph.edges[u, v]['msg'] = msg_out

    # Messages in backward phase
    for (u, v) in backward_path:  # Edge direction: u -> v
        msgs_in = graph.get_incoming_messages(u, exclude_node=v)
        msg_out = getattr(u, algorithm)(v, msgs_in)
        graph.edges[u, v]['msg'] = msg_out

    # Return belief
    msgs_in = graph.get_incoming_messages(query_node)
    return query_node.belief(msgs_in)


def _back_tracking(graph, query_node):
    """Back tracking.

    ...

    """

    # Depth First Search to determine edges
    # and convert tuple to a list
    dfs = nx.dfs_edges(graph, query_node)
    backward_path = list(dfs)

    # Back-tracking in backward phase
    track = {}

    msgs_in = graph.get_incoming_messages(query_node)
    track[query_node] = query_node.argmax(msgs_in)

    for (u, v) in backward_path:  # Edge direction: u -> v
        if v.type == nodes.NodeType.factor_node:
            for k in v.record[u].keys():
                track[k] = v.record[u][k][track[u]]

    return track


def belief_propagation(graph, query_node=None):
    """Belief propagation.

    Compute marginal distribution on graphs that are tree structured.

    """

    # Belief Propagation is equivalent to Sum-product algorithm.
    return sum_product(graph, query_node)


def sum_product(graph, query_node=None):
    """Sum-product algorithm.

    Compute marginal distribution on graphs that are tree structured.

    """

    if query_node is None:  # pick random node
        query_node = choice(graph.get_vnodes())

    # Return marginal probability for query node
    return _message_passing(graph, query_node, 'spa')


def max_product(graph, query_node=None):
    """Max-product algorithm.

    Compute maximum probability and setting of variables with maximum
    probability on graphs that are tree structured.

    """

    if query_node is None:  # pick random node
        query_node = choice(graph.get_vnodes())

    # Message passing
    max_prob = _message_passing(graph, query_node, 'mpa')

    # Back-tracking
    arg_max_prob = _back_tracking(graph, query_node)

    # Return maximum probability for query node and setting of variables
    return max_prob, arg_max_prob


def max_sum(graph, query_node=None):
    """Max-sum algorithm.

    Compute maximum probability and setting of variables with maximum
    probability on graphs that are tree structured.

    """

    if query_node is None:  # pick random node
        query_node = choice(graph.get_vnodes())

    # Message passing
    max_prob = _message_passing(graph, query_node, 'msa')

    # Back-tracking
    arg_max_prob = _back_tracking(graph, query_node)

    # Return maximum probability for query node and setting of variable
    return max_prob, arg_max_prob


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
