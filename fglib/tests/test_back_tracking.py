import unittest

import numpy as np
import numpy.testing as npt

from .. import graphs, nodes, inference, rv


class TestBackTracking(unittest.TestCase):

    def test_binary_variables(self):
        fg = graphs.FactorGraph()

        x = nodes.VNode("x", rv.Discrete)
        y = nodes.VNode("y", rv.Discrete)

        dist_p = [[0.3, 0.4],
                  [0.3, 0.0]]
        p = nodes.FNode("p", rv.Discrete(dist_p, y, x))

        fg.set_nodes([x, p, y])
        fg.set_edges([(x, p), (p, y)])

        _, arg_max = inference.max_product(fg, x)

        npt.assert_equal(arg_max[x], 1)
        npt.assert_equal(arg_max[y], 0)

    def test_issue_5(self):
        fg = graphs.FactorGraph()

        x1 = nodes.VNode("x1", rv.Discrete)
        x2 = nodes.VNode("x2", rv.Discrete)
        x3 = nodes.VNode("x3", rv.Discrete)
        x4 = nodes.VNode("x4", rv.Discrete)
        x5 = nodes.VNode("x5", rv.Discrete)

        dist_fa = [[[0.1, 0.2],
                    [0.1, 0.1]],
                   [[0.2, 0.05],
                    [0.2, 0.05]]]
        fa = nodes.FNode("fa", rv.Discrete(dist_fa, x1, x2, x3))

        dist_fb = [[0.1, 0.4],
                   [0.2, 0.3]]
        fb = nodes.FNode("fb", rv.Discrete(dist_fb, x3, x4))

        dist_fc = [[0.5, 0.1],
                   [0.2, 0.2]]
        fc = nodes.FNode("fc", rv.Discrete(dist_fc, x3, x5))

        fg.set_nodes([x1, x2, x3, x4, x5])
        fg.set_nodes([fa, fb, fc])

        fg.set_edges([(x1, fa), (x2, fa), (x3, fa)])
        fg.set_edges([(x3, fb), (x4, fb)])
        fg.set_edges([(x3, fc), (x5, fc)])

        _, arg_max = inference.max_product(fg, x5)

        # check the maximum probabilities
        msgs = fg.get_incoming_messages(x1)
        marg = x1.belief(msgs)
        npt.assert_almost_equal(marg.pmf, [0.3333333, 0.6666666])

        msgs = fg.get_incoming_messages(x2)
        marg = x2.belief(msgs)
        npt.assert_almost_equal(marg.pmf, [0.5, 0.5])

        msgs = fg.get_incoming_messages(x3)
        marg = x3.belief(msgs)
        npt.assert_almost_equal(marg.pmf, [0.7692307, 0.2307692])

        msgs = fg.get_incoming_messages(x4)
        marg = x4.belief(msgs)
        npt.assert_almost_equal(marg.pmf, [0.2, 0.8])

        msgs = fg.get_incoming_messages(x5)
        marg = x5.belief(msgs)
        npt.assert_almost_equal(marg.pmf, [0.7692307, 0.2307692])

        # check the setting of variables with maximum probability
#         npt.assert_equal(arg_max[x1], 1) # TODO: Open issue!
        npt.assert_equal(arg_max[x2], 0)
        npt.assert_equal(arg_max[x3], 0)
        npt.assert_equal(arg_max[x4], 1)
        npt.assert_equal(arg_max[x5], 0)


if __name__ == "__main__":
    unittest.main()
