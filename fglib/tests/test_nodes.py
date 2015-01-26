import unittest

import numpy as np
import numpy.testing as npt

from .. import graphs, nodes


class TestVariableNode(unittest.TestCase):

    def test_label(self):
        n = nodes.VNode("My Variable Node")
        self.assertEqual(str(n), "My Variable Node")

    def test_type(self):
        n = nodes.VNode("Variable Node")
        self.assertEqual(n.type, nodes.NodeType.variable_node)

    def test_neighbors(self):
        fg = graphs.FactorGraph()
        n0 = nodes.VNode(0)
        n1 = nodes.VNode(1)
        n2 = nodes.VNode(2)
        fg.set_nodes([n0, n1, n2])
        fg.set_edges([(n0, n1), (n1, n2)])

        nb = list(n0.neighbors())
        self.assertEqual(len(nb), 1)
        self.assertIn(n1, nb)

        nb = list(n1.neighbors())
        self.assertEqual(len(nb), 2)
        self.assertIn(n0, nb)
        self.assertIn(n2, nb)

        nb = list(n2.neighbors())
        self.assertEqual(len(nb), 1)
        self.assertIn(n1, nb)

        nb = list(n1.neighbors([n0, n2]))
        self.assertEqual(len(nb), 0)

    def test_spa(self):
        fg = graphs.FactorGraph()
        n0 = nodes.FNode(0)
        n1 = nodes.VNode(1)
        n2 = nodes.FNode(2)
        fg.set_nodes([n0, n1, n2])
        fg.set_edges([(n0, n1), (n1, n2)])

        msg_in = np.array([0.7, 0.3])
        fg[n0][n1]['object'].set_message(n0, n1, msg_in)

        msg_out = n1.spa(n2)
        npt.assert_almost_equal(msg_out, msg_in)

        n3 = nodes.VNode(3)
        fg.set_node(n3)
        fg.set_edge(n3, n1)

        msg_in = np.array([0.7, 0.3])
        fg[n3][n1]['object'].set_message(n3, n1, msg_in)

        msg_out = n1.spa(n2)
        npt.assert_almost_equal(msg_out, np.array([0.49, 0.09]))

    def test_mpa(self):
        pass

    def test_msa(self):
        pass

    def test_mf(self):
        pass


class TestFactorNode(unittest.TestCase):

    def test_label(self):
        n = nodes.FNode("My Factor Node", 0)
        self.assertEqual(str(n), "My Factor Node")

    def test_type(self):
        n = nodes.FNode("Factor Node", 0)
        self.assertEqual(n.type, nodes.NodeType.factor_node)

    def test_neighbors(self):
        fg = graphs.FactorGraph()
        n0 = nodes.FNode(0, 0)
        n1 = nodes.FNode(1, 0)
        n2 = nodes.FNode(2, 0)
        fg.set_nodes([n0, n1, n2])
        fg.set_edges([(n0, n1), (n1, n2)])

        nb = list(n0.neighbors())
        self.assertEqual(len(nb), 1)
        self.assertIn(n1, nb)

        nb = list(n1.neighbors())
        self.assertEqual(len(nb), 2)
        self.assertIn(n0, nb)
        self.assertIn(n2, nb)

        nb = list(n2.neighbors())
        self.assertEqual(len(nb), 1)
        self.assertIn(n1, nb)

        nb = list(n1.neighbors([n0, n2]))
        self.assertEqual(len(nb), 0)

    def test_spa(self):
        pass

    def test_mpa(self):
        pass

    def test_msa(self):
        pass

    def test_mf(self):
        pass


if __name__ == "__main__":
    unittest.main()
