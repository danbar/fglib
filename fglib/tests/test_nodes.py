import unittest

from .. import graphs
from .. import nodes
from fglib.nodes import NodeType


class TestNode(unittest.TestCase):

    def test_label(self):
        n = nodes.VNode("My Variable Node")
        self.assertEqual(str(n), "My Variable Node")

    def test_type(self):
        n = nodes.VNode("Variable Node")
        self.assertEqual(n.type, NodeType.variable_node)
        n = nodes.FNode("Factor Node", 0)
        self.assertEqual(n.type, NodeType.factor_node)

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


class TestEdge(unittest.TestCase):

    pass


class TestAlgorithm(unittest.TestCase):

    pass


if __name__ == "__main__":
    unittest.main()
