import unittest

import numpy as np
import numpy.testing as npt

from .. import nodes, rv


class TestDiscrete(unittest.TestCase):

    def setUp(self):
        self.x1 = nodes.VNode("x1", rv.Discrete)
        self.x2 = nodes.VNode("x2", rv.Discrete)

        self.rv1 = rv.Discrete([0.6, 0.4], self.x1)
        self.rv2 = rv.Discrete([0.2, 0.8], self.x2)
        self.rv3 = rv.Discrete([[0.1, 0.2],
                                [0.3, 0.4]], self.x1, self.x2)

    def test_equality(self):
        self.assertEqual(self.rv1, self.rv1)
        self.assertNotEqual(self.rv1, self.rv2)

    def test_initialization(self):
        with self.assertRaises(rv.ParameterException):
            x = nodes.VNode("x", rv.Discrete)
            rv.Discrete([[0.1, 0.2]], x)

    def test_string(self):
        s = str(self.rv1)
        self.assertEqual(s, '[ 0.6, 0.4]')

        s = str(self.rv3)
        self.assertEqual(s, '[[ 0.1, 0.2],\n [ 0.3, 0.4]]')

    @unittest.skip("Test case is not implemented.")
    def test_addition(self):
        # ToDo: ...
        pass

    @unittest.skip("Test case is not implemented.")
    def test_subtraction(self):
        # ToDo: ...
        pass

    def test_multiplication_1D_1(self):
        res = np.array([0.36, 0.16])
        res /= np.sum(res)

        mul = self.rv1 * self.rv1
        mul = mul.normalize()

        npt.assert_almost_equal(mul.pmf, res)
        self.assertEqual(mul.dim, (self.x1,))

    def test_multiplication_2D_1(self):
        res = np.array([[0.06, 0.12],
                        [0.12, 0.16]])
        res /= np.sum(res)

        mul = self.rv1 * self.rv3
        mul = mul.normalize()

        npt.assert_almost_equal(mul.pmf, res)
        self.assertEqual(mul.dim, (self.x1, self.x2))

    def test_multiplication_2D_2(self):
        res = np.array([[0.06, 0.12],
                        [0.12, 0.16]])
        res /= np.sum(res)

        mul = self.rv3 * self.rv1
        mul = mul.normalize()

        npt.assert_almost_equal(mul.pmf, res)
        self.assertEqual(mul.dim, (self.x1, self.x2))

    def test_multiplication_2D_3(self):
        res = np.array([[0.02, 0.16],
                        [0.06, 0.32]])
        res /= np.sum(res)

        mul = self.rv2 * self.rv3
        mul = mul.normalize()

        npt.assert_almost_equal(mul.pmf, res)
        self.assertEqual(mul.dim, (self.x1, self.x2))

    def test_multiplication_2D_4(self):
        res = np.array([[0.02, 0.16],
                        [0.06, 0.32]])
        res /= np.sum(res)

        mul = self.rv3 * self.rv2
        mul = mul.normalize()

        npt.assert_almost_equal(mul.pmf, res)
        self.assertEqual(mul.dim, (self.x1, self.x2))

    def test_unit_element_1D(self):
        rv0 = rv.Discrete.unity(self.x1)
        self.assertEqual(self.rv1 * rv0, self.rv1)

        rv0 = rv.Discrete([1, 1], self.x1)
        self.assertEqual(self.rv1 * rv0, self.rv1)

    def test_unit_element_2D(self):
        rv0 = rv.Discrete.unity(self.x1, self.x2)
        self.assertEqual(self.rv3 * rv0, self.rv3)

        rv0 = rv.Discrete([[1, 1],
                           [1, 1]], self.x1, self.x2)
        self.assertEqual(self.rv3 * rv0, self.rv3)

    def test_marginalize(self):
        res = np.array([0.4, 0.6])
        res /= np.sum(res)
        marginalize = self.rv3.marginalize(self.x1)
        npt.assert_almost_equal(marginalize.pmf, res)

        res = np.array([0.3, 0.7])
        res /= np.sum(res)
        marginalize = self.rv3.marginalize(self.x2)
        npt.assert_almost_equal(marginalize.pmf, res)

    def test_maximize(self):
        res = np.array([0.3, 0.4])
        res /= np.sum(res)
        amax = self.rv3.maximize(self.x1)
        npt.assert_almost_equal(amax.pmf, res)

        res = np.array([0.2, 0.4])
        res /= np.sum(res)
        amax = self.rv3.maximize(self.x2)
        npt.assert_almost_equal(amax.pmf, res)

    def test_argmax(self):
        self.assertEqual(self.rv1.argmax(), (0,))
        self.assertEqual(self.rv3.argmax(), (1, 1))
        self.assertEqual(self.rv3.argmax(self.x1), (1,))

    @unittest.skip("Test case is not implemented.")
    def test_log(self):
        pass


class TestGaussian(unittest.TestCase):

    def setUp(self):
        self.x1 = nodes.VNode("x1", rv.Gaussian)
        self.x2 = nodes.VNode("x2", rv.Gaussian)

        self.rv1 = rv.Gaussian([[1]], [[2]], self.x1)
        self.rv2 = rv.Gaussian([[3]], [[4]], self.x1)
        self.rv3 = rv.Gaussian([[1], [2]], [[3, 4], [5, 6]], self.x1, self.x2)
        self.rv4 = rv.Gaussian([[1], [4]], [[2, 0], [0, 8]], self.x1, self.x2)

    def test_equality(self):
        self.assertEqual(self.rv1, self.rv1)
        self.assertNotEqual(self.rv1, self.rv2)

    def test_initialization(self):
        mean = np.array([[1], [2]])
        cov = np.array([[3, 4], [5, 6]])
        npt.assert_almost_equal(self.rv3.mean, mean)
        npt.assert_almost_equal(self.rv3.cov, cov)

        tmp = rv.Gaussian.inf_form(np.linalg.inv(cov),
                                   np.dot(np.linalg.inv(cov), mean),
                                   self.x1, self.x2)
        self.assertEqual(self.rv3, tmp)

    def test_string(self):
        s = str(self.rv1)
        self.assertEqual(s, '[[ 1.]]\n[[ 2.]]')

        s = str(self.rv3)
        self.assertEqual(s, '[[ 1.],\n [ 2.]]\n[[ 3., 4.],\n [ 5., 6.]]')

    def test_addition_1D_1(self):
        add = self.rv1 + self.rv2
        res = rv.Gaussian([[4]], [[6]], self.x1)
        self.assertEqual(add, res)

        add += self.rv1
        res = rv.Gaussian([[5]], [[8]], self.x1)
        self.assertEqual(add, res)

    def test_subtraction_1D_1(self):
        sub = self.rv1 - self.rv2
        res = rv.Gaussian([[-2]], [[-2]], self.x1)
        self.assertEqual(sub, res)

        sub -= self.rv1
        res = rv.Gaussian([[-3]], [[-4]], self.x1)
        self.assertEqual(sub, res)

    def test_multiplication_1D_1(self):
        mul = self.rv2 * self.rv2
        res = rv.Gaussian([[3]], [[2]], self.x1)
        self.assertEqual(mul, res)

        mul *= self.rv1
        res = rv.Gaussian([[2]], [[1]], self.x1)
        self.assertEqual(mul, res)

    def test_unit_element_1D(self):
        rv0 = rv.Gaussian.unity(self.x1)
        self.assertEqual(self.rv1 * rv0, self.rv1)

        rv0 = rv.Gaussian([[0]], [[np.Inf]], self.x1)
        self.assertEqual(self.rv1 * rv0, self.rv1)

    def test_unit_element_2D(self):
        rv0 = rv.Gaussian.unity(self.x1, self.x2)
        self.assertEqual(self.rv3 * rv0, self.rv3)

        rv0 = rv.Gaussian([[0], [0]],
                          [[np.Inf, 0], [0, np.Inf]],
                          self.x1, self.x2)
        self.assertEqual(self.rv3 * rv0, self.rv3)

    def test_marginalize(self):
        res = self.rv1
        marginalize = self.rv4.marginalize(self.x2)
        self.assertEqual(marginalize, res)

    def test_maximize(self):
        res = self.rv1
        amax = self.rv1.maximize()
        self.assertEqual(amax, res)

        res = self.rv4
        amax = self.rv4.maximize()
        self.assertEqual(amax, res)

    def test_argmax(self):
        npt.assert_almost_equal(self.rv1.argmax(), self.rv1.mean)
        npt.assert_almost_equal(self.rv3.argmax(), self.rv3.mean)
        npt.assert_almost_equal(self.rv3.argmax(self.x2),
                                np.atleast_2d(self.rv3.mean[0]))

    @unittest.skip("Test case is not implemented.")
    def test_log(self):
        pass


if __name__ == "__main__":
    unittest.main()
