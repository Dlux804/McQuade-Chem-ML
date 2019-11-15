import unittest



class TestSum(unittest.TestCase):
    def test_sample(self):
        """
        Test that it can do a test
        """
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)

if __name__ == '__main__':
    unittest.main()
