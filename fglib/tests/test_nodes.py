import unittest

import numpy as np
import numpy.testing as npt

from .. import graphs, nodes, rv


class TestVariableNode(unittest.TestCase):

    def test_label(self):
        n = nodes.VNode("My Variable Node", rv.Gaussian)
        self.assertEqual(str(n), "My Variable Node")

    def test_type(self):
        n = nodes.VNode("Variable Node", rv.Discrete)
        self.assertEqual(n.type, nodes.NodeType.variable_node)

    def test_neighbors(self):
        fg = graphs.FactorGraph()
        n0 = nodes.VNode(0, rv.Discrete)
        n1 = nodes.VNode(1, rv.Discrete)
        n2 = nodes.VNode(2, rv.Discrete)
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
        n1 = nodes.VNode(1, rv.Discrete)
        n2 = nodes.FNode(2)
        fg.set_nodes([n0, n1, n2])
        fg.set_edges([(n0, n1), (n1, n2)])

        msg_in = rv.Discrete(np.array([0.7, 0.3]), n1)
        fg[n0][n1]['object'].set_message(n0, n1, msg_in)

        msg_out = n1.spa(n2)
        msg_out = msg_out.normalize()

        npt.assert_almost_equal(msg_out.pmf, msg_in.pmf)
        self.assertEqual(msg_out.dim, (n1,))

        n3 = nodes.VNode(3, rv.Discrete)
        fg.set_node(n3)
        fg.set_edge(n3, n1)

        msg_in = rv.Discrete(np.array([0.7, 0.3]), n1)
        fg[n3][n1]['object'].set_message(n3, n1, msg_in)

        msg_out = n1.spa(n2)
        msg_out = msg_out.normalize()

        res = np.array([0.49, 0.09])
        res /= np.sum(res)
        npt.assert_almost_equal(msg_out.pmf, res)
        self.assertEqual(msg_out.dim, (n1,))

    @unittest.skip("Test case is not implemented.")
    def test_mpa(self):
        pass

    @unittest.skip("Test case is not implemented.")
    def test_msa(self):
        pass

    @unittest.skip("Test case is not implemented.")
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
        fg = graphs.FactorGraph()
        n0 = nodes.VNode(0, rv.Discrete)
        n2 = nodes.VNode(2, rv.Discrete)
        rv1 = rv.Discrete(np.array([[0.3, 0.4],
                                    [0.3, 0.0]]), n0, n2)
        n1 = nodes.FNode(1, rv1)
        fg.set_nodes([n0, n1, n2])
        fg.set_edges([(n0, n1), (n1, n2)])

        tmp = np.array([0.49, 0.09])
        tmp /= np.sum(tmp)
        msg_in = rv.Discrete(tmp, n2)
        fg[n2][n1]['object'].set_message(n2, n1, msg_in)

        msg_out = n1.spa(n0)
        msg_out = msg_out.normalize()

        res = np.array([0.183, 0.147])
        res /= np.sum(res)
        npt.assert_almost_equal(msg_out.pmf, res)
        self.assertEqual(msg_out.dim, (n0,))

    @unittest.skip("Test case is not implemented.")
    def test_mpa(self):
        pass

    @unittest.skip("Test case is not implemented.")
    def test_msa(self):
        pass

    @unittest.skip("Test case is not implemented.")
    def test_mf(self):
        pass


if __name__ == "__main__":
    unittest.main()
