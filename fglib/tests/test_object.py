import unittest

from .. import model
from .. import object


class TestNode(unittest.TestCase):

    def test_label(self):
        n = object.VNode("My Variable Node")
        self.assertEqual(str(n), "My Variable Node")

    def test_type(self):
        n = object.VNode("Variable Node")
        self.assertEqual(n.TYPE, "vn")
        n = object.FNode("Factor Node", 0)
        self.assertEqual(n.TYPE, "fn")
        self.assertEqual(n.factor, 0)

    def test_neighbors(self):
        fg = model.FactorGraph()
        n0 = object.VNode(0)
        n1 = object.VNode(1)
        n2 = object.VNode(2)
        fg.set_nodes([n0, n1, n2])
        fg.set_edges([(n0, n1), (n1, n2)])

        nb = list(n0.neighbors(fg, n0))
        self.assertEqual(len(nb), 1)
        self.assertIn(n1, nb)

        nb = list(n0.neighbors(fg, n1))
        self.assertEqual(len(nb), 2)
        self.assertIn(n0, nb)
        self.assertIn(n2, nb)

        nb = list(n0.neighbors(fg, n1, n2))
        self.assertEqual(len(nb), 1)
        self.assertIn(n0, nb)
        self.assertNotIn(n2, nb)

        nb = list(n0.neighbors(fg, n1, [n0, n2]))
        self.assertEqual(len(nb), 0)
        self.assertNotIn(n0, nb)
        self.assertNotIn(n2, nb)


class TestEdge(unittest.TestCase):

    pass


class TestAlgorithm(unittest.TestCase):

    pass


if __name__ == "__main__":
    unittest.main()
