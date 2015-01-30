#!/usr/bin/env python

"""A simple example

This simple example of the sum-product algorithm on a factor graph
with Bernoulli random variables is taken from page 409 of the book
C. M. Bishop, Pattern Recognition and Machine Learning. Springer, 2006.

      /--\      +----+      /--\      +----+      /--\
     | x1 |-----| fa |-----| x2 |-----| fb |-----| x3 |
      \--/      +----+      \--/      +----+      \--/
                             |
                           +----+
                           | fc |
                           +----+
                             |
                            /--\
                           | x4 |
                            \--/

"""

from fglib import graphs, nodes, inference, rv

# Create factor graph
fg = graphs.FactorGraph()

# Create variable nodes
x1 = nodes.VNode("x1")
x2 = nodes.VNode("x2")
x3 = nodes.VNode("x3")
x4 = nodes.VNode("x4")

# Create factor nodes
fa = nodes.FNode("fa")
fb = nodes.FNode("fb")
fc = nodes.FNode("fc")

# Add nodes to factor graph
fg.set_nodes([x1, x2, x3, x4])
fg.set_nodes([fa, fb, fc])

# Add edges to factor graph
fg.set_edge(x1, fa)
fg.set_edge(fa, x2)
fg.set_edge(x2, fb)
fg.set_edge(fb, x3)
fg.set_edge(x2, fc)
fg.set_edge(fc, x4)

# Set joint distributions of factor nodes over variable nodes
dist_fa = [[0.3, 0.4],
           [0.3, 0.0]]
fa.factor = rv.Discrete(dist_fa, x1, x2)

dist_fb = [[0.3, 0.4],
           [0.3, 0.0]]
fb.factor = rv.Discrete(dist_fb, x2, x3)

dist_fc = [[0.3, 0.4],
           [0.3, 0.0]]
fc.factor = rv.Discrete(dist_fc, x2, x4)

# Perform sum-product algorithm on factor graph
# and request belief of variable node x3
belief = inference.sum_product(fg, x3)

# Print belief of variable node x3
print("Belief of variable node x3:")
print(belief)
