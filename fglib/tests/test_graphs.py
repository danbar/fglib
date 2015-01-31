import unittest

import numpy as np
import numpy.testing as npt

from .. import graphs, nodes, inference, rv


class TestFactorGraph(unittest.TestCase):

    def setUp(self):
        # Create factor graph
        self.fg = graphs.FactorGraph()

        # Create variable nodes
        self.x1 = nodes.VNode("x1")
        self.x2 = nodes.VNode("x2")
        self.x3 = nodes.VNode("x3")
        self.x4 = nodes.VNode("x4")

        # Create factor nodes
        self.fa = nodes.FNode("fa")
        self.fb = nodes.FNode("fb")
        self.fc = nodes.FNode("fc")

        # Add nodes to factor graph
        self.fg.set_nodes([self.x1, self.x2, self.x3, self.x4])
        self.fg.set_nodes([self.fa, self.fb, self.fc])

        # Add edges to factor graph
        self.fg.set_edge(self.x1, self.fa)
        self.fg.set_edge(self.fa, self.x2)
        self.fg.set_edge(self.x2, self.fb)
        self.fg.set_edge(self.fb, self.x3)
        self.fg.set_edge(self.x2, self.fc)
        self.fg.set_edge(self.fc, self.x4)

        # Set joint distributions of factor nodes over variable nodes
        dist_fa = [[0.3, 0.4],
                   [0.3, 0.0]]
        self.fa.factor = rv.Discrete(dist_fa, self.x1, self.x2)

        dist_fb = [[0.3, 0.4],
                   [0.3, 0.0]]
        self.fb.factor = rv.Discrete(dist_fb, self.x2, self.x3)

        dist_fc = [[0.3, 0.4],
                   [0.3, 0.0]]
        self.fc.factor = rv.Discrete(dist_fc, self.x2, self.x4)

    def test_spa(self):
        inference.sum_product(self.fg)

        # Test belief of variable node x1
        belief = self.x1.belief()
        res = np.array([0.183, 0.147])
        res /= np.sum(res)
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.x1,))

        # Test belief of variable node x2
        belief = self.x2.belief()
        res = np.array([0.294, 0.036])
        res /= np.sum(res)
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.x2,))

        # Test belief of variable node x3
        belief = self.x3.belief()
        res = np.array([0.162, 0.168])
        res /= np.sum(res)
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.x3,))

        # Test belief of variable node x4
        belief = self.x4.belief()
        res = np.array([0.162, 0.168])
        res /= np.sum(res)
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.x4,))

    @unittest.skip("Test case is not implemented.")
    def test_mpa(self):
        pass

    @unittest.skip("Test case is not implemented.")
    def test_msa(self):
        pass


if __name__ == "__main__":
    unittest.main()
