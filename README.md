[![Build Status](https://www.travis-ci.org/danbar/fglib.svg?branch=master)](https://www.travis-ci.org/danbar/fglib)

![logo of fglib](https://rawgit.com/danbar/fglib/master/docs/logo.svg)

# fglib

The factor graph library (fglib) is a Python package to simulate message passing on factor graphs.
It supports the

* sum-product algorithm (belief propagation)
* max-product algorithm
* max-sum algorithm
* mean-field algorithm

with discrete and Gaussian random variables.

This Python package is build upon the Python packages [NetworkX](https://networkx.github.io/) and [NumPy](http://www.numpy.org/).

## Dependencies

The following dependencies are required to run fglib:

* _Python_ 3.4 or later
* _NetworkX_ 2.0 or later
* _NumPy_ 1.12 or later
* _matplotlib_ 2.0 or later

In addition, the Python package _setuptools_ is required to install fglib.

## Documentation

In order to generate the documentation site for the factor graph library, execute the following commands from the top-level directory.

```
$ cd docs/
$ make html
```

## Example

```Python
"""A simple example of the sum-product algorithm

This is a simple example of the sum-product algorithm on a factor graph
with Discrete random variables.

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

     fa   | x2=0 x2=1 x2=2     fb   | x3=0 x3=1     fc   | x4=0 x4=1
     ---------------------     ----------------     ----------------
     x1=0 | 0.3  0.2  0.1      x2=0 | 0.3  0.2      x2=0 | 0.3  0.2
     x1=1 | 0.3  0.0  0.1      x2=1 | 0.3  0.0      x2=1 | 0.3  0.0
                               x2=2 | 0.1  0.1      x2=2 | 0.1  0.1

"""

from fglib import graphs, nodes, inference, rv

# Create factor graph
fg = graphs.FactorGraph()

# Create variable nodes
x1 = nodes.VNode("x1", rv.Discrete)  # with 2 states (Bernoulli)
x2 = nodes.VNode("x2", rv.Discrete)  # with 3 states
x3 = nodes.VNode("x3", rv.Discrete)
x4 = nodes.VNode("x4", rv.Discrete)

# Create factor nodes (with joint distributions)
dist_fa = [[0.3, 0.2, 0.1],
           [0.3, 0.0, 0.1]]
fa = nodes.FNode("fa", rv.Discrete(dist_fa, x1, x2))

dist_fb = [[0.3, 0.2],
           [0.3, 0.0],
           [0.1, 0.1]]
fb = nodes.FNode("fb", rv.Discrete(dist_fb, x2, x3))

dist_fc = [[0.3, 0.2],
           [0.3, 0.0],
           [0.1, 0.1]]
fc = nodes.FNode("fc", rv.Discrete(dist_fc, x2, x4))

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

# Perform sum-product algorithm on factor graph
# and request belief of variable node x4
belief = inference.sum_product(fg, x4)

# Print belief of variables
print("Belief of variable node x4:")
print(belief)
```

## References

* H.-A. Loeliger, “An introduction to factor graphs,” 
_IEEE Signal Process. Mag._, vol. 21, no. 1, pp. 28–41, Jan. 2004.

* F. R. Kschischang, B. J. Frey, and H.-A. Loeliger, “Factor graphs and the sum-product algorithm,” 
_IEEE Trans. Inf. Theory_, vol. 47, no. 2, pp. 498–519, Feb. 2001.

* H.-A. Loeliger, J. Dauwels, H. Junli, S. Korl, P. Li, and F. R. Kschischang, “The factor graph approach to model-based signal processing,” 
_Proc. IEEE_, vol. 95, no. 6, pp. 1295–1322, Jun. 2007.

* H. Wymeersch, _Iterative Receiver Design_.
Cambridge, UK: Cambridge University Press, 2007.

* C. M. Bishop, _Pattern Recognition and Machine Learning_, 
8th ed., ser. Information Science and Statistics.
New York, USA: Springer Science+Business Media, 2009.
