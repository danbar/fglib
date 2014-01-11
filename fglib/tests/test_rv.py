import unittest
import numpy as np
import numpy.testing as npt

from .. import rv


class TestDiscrete(unittest.TestCase):

    def setUp(self):
        self.d1 = rv.Discrete([0.6, 0.4], [0])
        self.d2 = rv.Discrete([0.2, 0.8], [1])
        self.d3 = rv.Discrete([[0.1, 0.2], [0.3, 0.4]], [0, 1])

    def test_equality(self):
        self.assertEqual(self.d1, self.d1)
        self.assertNotEqual(self.d1, self.d2)

    def test_initialization(self):
        with self.assertRaises(rv.ParameterException):
            rv.Discrete([[0.1, 0.2]], [0])

    @unittest.skip("Test case is not implemented.")
    def test_addition(self):
        #TODO: ...
        pass

    @unittest.skip("Test case is not implemented.")
    def test_subtraction(self):
        #TODO: ...
        pass

    def test_multiplication(self):
        tmp = np.array([0.36, 0.16])
        npt.assert_almost_equal((self.d1 * self.d1).pmf, tmp / np.sum(tmp))
        tmp = np.array([[0.06, 0.12], [0.12, 0.16]])
        npt.assert_almost_equal((self.d1 * self.d3).pmf, tmp / np.sum(tmp))
        tmp = np.array([[0.02, 0.16], [0.06, 0.32]])
        npt.assert_almost_equal((self.d2 * self.d3).pmf, tmp / np.sum(tmp))
        tmp = np.array([[0.01, 0.04], [0.09, 0.16]])
        npt.assert_almost_equal((self.d3 * self.d3).pmf, tmp / np.sum(tmp))

    def test_unit_element(self):
        ue = rv.Discrete([0.5, 0.5], [0])
        self.assertEqual(self.d1 * ue, self.d1)
        ue = rv.Discrete([[0.25, 0.25], [0.25, 0.25]], [0, 1])
        self.assertEqual(self.d3 * ue, self.d3)

    def test_argmax(self):
        self.assertEqual(self.d1.argmax(), (0,))
        self.assertEqual(self.d3.argmax(), (1, 1))
        self.assertEqual(self.d3.argmax((0,)), (1,))

    def test_max(self):
        self.assertEqual(self.d1.max(), 0.6)
        self.assertEqual(self.d3.max(), 0.4)
        npt.assert_almost_equal(self.d3.max((0,)), 0.7)

    def test_marginal(self):
        tmp = np.array([0.3, 0.7])
        npt.assert_almost_equal(self.d3.marginal((0,)).pmf, tmp)
        tmp = np.array([0.4, 0.6])
        npt.assert_almost_equal(self.d3.marginal((1,)).pmf, tmp)

    @unittest.skip("Method is not implemented.")
    def test_log(self):
        pass


class TestGaussian(unittest.TestCase):

    def setUp(self):
        self.g1 = rv.Gaussian([[1]], [[2]])
        self.g2 = rv.Gaussian([[3]], [[4]])
        self.g3 = rv.Gaussian([[1], [2]], [[3, 4], [5, 6]])
        self.g4 = rv.Gaussian([[1], [4]], [[2, 0], [0, 8]])

    def test_equality(self):
        self.assertEqual(self.g1, self.g1)
        self.assertNotEqual(self.g1, self.g2)

    def test_initialization(self):
        mean = np.array([[1], [2]])
        cov = np.array([[3, 4], [5, 6]])
        npt.assert_almost_equal(self.g3.mean, mean)
        npt.assert_almost_equal(self.g3.cov, cov)
        tmp = rv.Gaussian.moment_form(mean, cov)
        self.assertEqual(self.g3, tmp)
        tmp = rv.Gaussian.information_form(np.linalg.inv(cov),
                                           np.dot(np.linalg.inv(cov), mean))
        self.assertEqual(self.g3, tmp)

    def test_addition(self):
        tmp = self.g1 + self.g2
        self.assertEqual(tmp, rv.Gaussian([[4]], [[6]]))
        tmp += self.g1
        self.assertEqual(tmp, rv.Gaussian([[5]], [[8]]))

    def test_subtraction(self):
        tmp = self.g1 - self.g2
        self.assertEqual(tmp, rv.Gaussian([[-2]], [[-2]]))
        tmp -= self.g1
        self.assertEqual(tmp, rv.Gaussian([[-3]], [[-4]]))

    def test_multiplication(self):
        tmp = self.g2 * self.g2
        self.assertEqual(tmp, rv.Gaussian([[3]], [[2]]))
        tmp *= self.g1
        self.assertEqual(tmp, rv.Gaussian([[2]], [[1]]))

    def test_unit_element(self):
        ue = rv.Gaussian([[0]], [[np.Inf]])
        self.assertEqual(self.g1 * ue, self.g1)
        ue = rv.Gaussian([[0], [0]], [[np.Inf, 0], [0, np.Inf]])
        self.assertEqual(self.g3 * ue, self.g3)

    def test_argmax(self):
        npt.assert_almost_equal(self.g1.argmax(), np.array([[1]]))
        npt.assert_almost_equal(self.g3.argmax(), np.array([[1], [2]]))
        npt.assert_almost_equal(self.g3.argmax([0]), np.array([[1]]))
        npt.assert_almost_equal(self.g3.argmax([0, 1]), np.array([[1], [2]]))

    def test_max(self):
        self.assertEqual(self.g1.max(), np.sqrt(4 * np.pi))
        self.assertEqual(self.g4.max(),
                         2 * np.pi * np.sqrt(np.linalg.det(self.g4.cov)))

    def test_marginal(self):
        self.assertEqual(self.g4.marginal([0]), self.g1)

    @unittest.skip("Method is not implemented.")
    def test_log(self):
        pass


if __name__ == "__main__":
    unittest.main()
