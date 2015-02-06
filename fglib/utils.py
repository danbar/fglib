"""Module for utilities.

This module contains auxiliary functions for factor graphs.

Functions:
    draw: Draw a factor graph with nodes, edges and labels.
    draw_message: Draw messages of a factor graph.
    draw_attribute: Draw node attributes of a factor graph.

"""

import matplotlib.pyplot as plt
import networkx as nx


def draw(graph, pos=None):
    """Draw factor graph and return used positions for nodes."""
    if pos is None:
        pos = nx.spring_layout(graph)

    # Draw variable nodes
    vn = [n for (n, d) in graph.nodes(data=True) if d['type'] == "vn"]

    vn_observed = [n for n in vn if n.observed]
    nx.draw_networkx_nodes(graph, pos, nodelist=vn_observed, node_size=1000,
                           node_color="gray", node_shape='o')

    vn_hidden = [n for n in vn if not n.observed]
    nx.draw_networkx_nodes(graph, pos, nodelist=vn_hidden, node_size=1000,
                           node_color="white", node_shape='o')

    # Draw factor nodes
    fn = [n for (n, d) in graph.nodes(data=True) if d['type'] == "fn"]
    nx.draw_networkx_nodes(graph, pos, nodelist=fn, node_size=1500,
                           node_color="white", node_shape='s')

    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=22, font_family='sans-serif')

    # Draw edges
    nx.draw_networkx_edges(graph, pos, alpha=0.5, edge_color='black')

    return pos


def draw_message(graph, pos):
    """Draw messages of a factor graph."""
    msg = {}  # Dict of node tuples to edge labels: {(nodeX, nodeY): aString}
    for u, v in graph.edges():
        m = graph.get_edge_data(u, v)["object"]
        s = "$m_{" + str(u).replace('$', '') + " -> " + \
            str(v).replace('$', '') + "}$ = " + str(m.get_message(u, v))
        s = s + "\n\n"
        s = s + "$m_{" + str(v).replace('$', '') + " -> " + \
            str(u).replace('$', '') + "}$ = " + str(m.get_message(v, u))
        msg[(u, v)] = s
    bbox_props = dict(boxstyle='round',
                      alpha=0.5,
                      ec="none",
                      fc="none")
    nx.draw_networkx_edge_labels(graph, pos, font_size=12, edge_labels=msg,
                                 bbox=bbox_props)  # draw the edge labels


def draw_attribute(graph, pos, attr):
    """Draw node attributes of a factor graph."""
    labels = dict((n, d[attr]) for n, d in graph.nodes(data=True) if attr in d)
    for n, d in labels.items():
        x, y = pos[n]
        plt.text(x, y - 0.1, s="%s = %s" % (attr, d),
                 bbox=dict(facecolor='red', alpha=0.5),
                 horizontalalignment='center')
