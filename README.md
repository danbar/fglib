![logo of fglib](https://rawgit.com/danbar/fglib/master/docs/logo.svg)

fglib
=====

fglib (factor graph library) is a Python 3 package to simulate message passing on factor graphs.

The project is in the development phase.

Dependencies
------------

* NetworkX 1.7 or later
* NumPy 1.8 or later
* matplotlib 1.3 or later

Example
-------

```Python

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

The following joint distributions are used for the factor nodes.

     fa   | x2=0 x2=1     fb   | x3=0 x3=1     fc   | x4=0 x4=1
     ----------------     ----------------     ----------------
     x1=0 | 0.3  0.4      x2=0 | 0.3  0.4      x2=0 | 0.3  0.4
     x1=1 | 0.3  0.0      x2=1 | 0.3  0.0      x2=1 | 0.3  0.0

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

```

References
----------

* H.-A. Loeliger, “An introduction to factor graphs,” 
_IEEE Signal Process. Mag._, vol. 21, no. 1, pp. 28–41, Jan. 2004.
* F. R. Kschischang, B. J. Frey, and H.-A. Loeliger, “Factor graphs and the sum-product algorithm,” 
_IEEE Trans. Inf. Theory_, vol. 47, no. 2, pp. 498–519, Feb. 2001.
* H.-A. Loeliger, J. Dauwels, H. Junli, S. Korl, P. Li, and F. R. Kschischang, “The factor graph approach to model-based signal processing,” 
_Proc. IEEE_, vol. 95, no. 6, pp. 1295–1322, Jun. 2007.
* C. M. Bishop, _Pattern Recognition and Machine Learning_, 
8th ed., ser. Information Science and Statistics. New York, USA: Springer Science+Business Media, 2009.
