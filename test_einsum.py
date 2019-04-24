import unittest
import numpy as np
from einsum import einsum


class TestEinsum(unittest.TestCase):
    def test_einsum(self):
        np.random.seed(0)
        A = np.random.random((6, 8, 6))
        B = np.random.random((8, 10, 2))

        E1 = np.einsum("iji,jkl -> jkl", A, B)
        E2 = einsum("i1,i2,i1; i2,i3,i4 -> i2,i3,i4", A, B)
        self.assertTrue(np.allclose(E1, E2))


if __name__ == "__main__":
    unittest.main()
