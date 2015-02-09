import unittest

import networkx as nx
from networkx.algorithms import bipartite

from .. import graphs, nodes


class TestFactorGraph(unittest.TestCase):

    def test_convert_graph_to_factor_garph(self):
        g = nx.Graph()
        g.add_nodes_from(['x1', 'x2', 'x3', 'x4'], bipartite=0)
        g.add_nodes_from(['fa', 'fb', 'fc'], bipartite=1)
        g.add_edges_from([('x1', 'fa'), ('fa', 'x2'),
                          ('x2', 'fb'), ('fb', 'x3'),
                          ('x2', 'fc'), ('fc', 'x2')])
        bottom_nodes, top_nodes = bipartite.sets(g)

        fg = graphs.convert_graph_to_factor_graph(g, nodes.VNode, nodes.FNode)
        vn = {str(n) for n in fg.get_vnodes()}
        fn = {str(n) for n in fg.get_fnodes()}

        self.assertSetEqual(top_nodes, vn)
        self.assertSetEqual(bottom_nodes, fn)


if __name__ == "__main__":
    unittest.main()
