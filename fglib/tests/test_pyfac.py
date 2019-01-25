"""Test cases from pyfac package.

Test fglib with the test cases from the pyfac package,
which is written by Ryan Lester [1].

[1] https://github.com/rdlester/pyfac

"""

import unittest

import numpy as np
import numpy.testing as npt

from .. import graphs, nodes, inference, rv


class TestPyfac(unittest.TestCase):

    def test_toy_graph(self):
        # Create toy graph:
        #
        # Simple graph encoding, basic testing
        # 2 vars, 2 facs
        # f_a, f_ba - p(a)p(a|b)
        # factors functions are a little funny but it works
        G = graphs.FactorGraph()

        a = nodes.VNode('a', rv.Discrete)  # dim = 3
        b = nodes.VNode('b', rv.Discrete)  # dim = 2

        Pb = [0.3, 0.7]
        Pb = nodes.FNode('Pb', rv.Discrete(Pb, b))

        Pab = [[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]]
        Pab = nodes.FNode('Pab', rv.Discrete(Pab, a, b))

        G.set_nodes([a, b, Pb, Pab])
        G.set_edges([(b, Pb), (a, Pab), (b, Pab)])

        inference.sum_product(G)

        # check the results
        # want to verify incoming messages
        # if vars are correct then factors must be as well
        msg = G.get_message(Pab, a)
        npt.assert_almost_equal(msg.pmf[0], 0.34065934)
        npt.assert_almost_equal(msg.pmf[1], 0.2967033)
        npt.assert_almost_equal(msg.pmf[2], 0.36263736)

        msg = G.get_message(Pb, b)
        npt.assert_almost_equal(msg.pmf[0], 0.3)
        npt.assert_almost_equal(msg.pmf[1], 0.7)

        msg = G.get_message(Pab, b)
        npt.assert_almost_equal(msg.pmf[0], 0.23333333)
        npt.assert_almost_equal(msg.pmf[1], 0.76666667)

        # check the marginals
        msgs = G.get_incoming_messages(a)
        marg = a.belief(msgs)
        npt.assert_almost_equal(marg.pmf[0], 0.34065934)
        npt.assert_almost_equal(marg.pmf[1], 0.2967033)
        npt.assert_almost_equal(marg.pmf[2], 0.36263736)

        msgs = G.get_incoming_messages(b)
        marg = b.belief(msgs)
        npt.assert_almost_equal(marg.pmf[0], 0.11538462)
        npt.assert_almost_equal(marg.pmf[1], 0.88461538)

    def test_test_graph(self):
        # Graph for HW problem 1.c.
        # 4 vars, 3 facs
        # f_a, f_ba, f_dca
        G = graphs.FactorGraph()

        a = nodes.VNode('a', rv.Discrete)  # dim = 2
        b = nodes.VNode('b', rv.Discrete)  # dim = 3
        c = nodes.VNode('c', rv.Discrete)  # dim = 4
        d = nodes.VNode('d', rv.Discrete)  # dim = 5

        p = np.array([0.3, 0.7])
        Pa = nodes.FNode('Pa', rv.Discrete(p, a))

        p = np.array([[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
        Pba = nodes.FNode('Pba', rv.Discrete(p, b, a))

        p = np.array([[[3., 1.], [1.2, 0.4], [0.1, 0.9], [0.1, 0.9]],
                      [[11., 9.], [8.8, 9.4], [6.4, 0.1], [8.8, 9.4]],
                      [[3., 2.], [2., 2.], [2., 2.], [3., 2.]],
                      [[0.3, 0.7], [0.44, 0.56], [0.37, 0.63], [0.44, 0.56]],
                      [[0.2, 0.1], [0.64, 0.44], [0.37, 0.63], [0.2, 0.1]]])
        Pdca = nodes.FNode('Pdca', rv.Discrete(p, d, c, a))

        # add a loop - not a part of 1.c., just for testing
        # p = np.array([[0.3, 0.2214532], [0.1, 0.4] ,
        #               [0.33333, 0.76], [0.1, 0.98]])
        # G.addFacNode(p, c, a)

        G.set_nodes([a, b, c, d, Pa, Pba, Pdca])
        G.set_edges([(a, Pa),
                     (b, Pba), (a, Pba),
                     (d, Pdca), (c, Pdca), (a, Pdca)])

        inference.sum_product(G)

        # check the marginals
        msgs = G.get_incoming_messages(a)
        marg = a.belief(msgs)
        npt.assert_almost_equal(marg.pmf[0], 0.13755539)
        npt.assert_almost_equal(marg.pmf[1], 0.86244461)

        msgs = G.get_incoming_messages(b)
        marg = b.belief(msgs)
        npt.assert_almost_equal(marg.pmf[0], 0.33928227)
        npt.assert_almost_equal(marg.pmf[1], 0.30358863)
        npt.assert_almost_equal(marg.pmf[2], 0.3571291)

        msgs = G.get_incoming_messages(c)
        marg = c.belief(msgs)
        npt.assert_almost_equal(marg.pmf[0], 0.30378128)
        npt.assert_almost_equal(marg.pmf[1], 0.29216947)
        npt.assert_almost_equal(marg.pmf[2], 0.11007584)
        npt.assert_almost_equal(marg.pmf[3], 0.29397341)

        msgs = G.get_incoming_messages(d)
        marg = d.belief(msgs)
        npt.assert_almost_equal(marg.pmf[0], 0.076011)
        npt.assert_almost_equal(marg.pmf[1], 0.65388724)
        npt.assert_almost_equal(marg.pmf[2], 0.18740039)
        npt.assert_almost_equal(marg.pmf[3], 0.05341787)
        npt.assert_almost_equal(marg.pmf[4], 0.0292835)


if __name__ == "__main__":
    unittest.main()
