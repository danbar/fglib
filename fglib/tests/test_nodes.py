import unittest

import numpy as np
import numpy.testing as npt

from .. import nodes, rv


class TestVariableNode(unittest.TestCase):

    def test_label(self):
        n = nodes.VNode("My Variable Node", rv.Gaussian)
        self.assertEqual(str(n), "My Variable Node")
        self.assertEqual(n.label, "My Variable Node")

    def test_type(self):
        n = nodes.VNode("Variable Node", rv.Discrete)
        self.assertEqual(n.type, nodes.NodeType.variable_node)

    def test_spa(self):
        # single input - single output
        n0 = nodes.VNode(1, rv.Discrete)
        n1 = nodes.FNode(2)

        msg_in0 = rv.Discrete(np.array([0.7, 0.3]), n0)

        msg_out = n0.spa(n1, [msg_in0])
        msg_out = msg_out.normalize()

        npt.assert_almost_equal(msg_out.pmf, msg_in0.pmf)
        self.assertEqual(msg_out.dim, (n0,))

        # multiple input - single output
        msg_in1 = rv.Discrete(np.array([0.7, 0.3]), n0)

        msg_out = n0.spa(n1, [msg_in0, msg_in1])
        msg_out = msg_out.normalize()

        res = np.array([0.49, 0.09])
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


class TestFactorNode(unittest.TestCase):

    def test_label(self):
        n = nodes.FNode("My Factor Node", 0)
        self.assertEqual(str(n), "My Factor Node")
        self.assertEqual(n.label, "My Factor Node")

    def test_type(self):
        n = nodes.FNode("Factor Node", 0)
        self.assertEqual(n.type, nodes.NodeType.factor_node)

    def test_spa(self):
        # single input - single output
        n0 = nodes.VNode(0, rv.Discrete)
        n1 = nodes.VNode(2, rv.Discrete)
        rv1 = rv.Discrete(np.array([[0.3, 0.4],
                                    [0.3, 0.0]]), n0, n1)
        n2 = nodes.FNode(1, rv1)

        tmp = np.array([0.7, 0.3])
        msg_in = rv.Discrete(tmp, n1)

        msg_out = n2.spa(n0, [msg_in])
        msg_out = msg_out.normalize()

        res = np.array([0.33, 0.21])
        res /= np.sum(res)
        npt.assert_almost_equal(msg_out.pmf, res)
        self.assertEqual(msg_out.dim, (n0,))

        tmp = np.array([0.1, 0.9])
        msg_in = rv.Discrete(tmp, n0)

        msg_out = n2.spa(n1, [msg_in])
        msg_out = msg_out.normalize()

        res = np.array([0.3, 0.04])
        res /= np.sum(res)
        npt.assert_almost_equal(msg_out.pmf, res)
        self.assertEqual(msg_out.dim, (n1,))

        # TODO: multiple input - single output

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
