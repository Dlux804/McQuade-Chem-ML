def some_function(x):
    return x+1

class TestClass:

    def test_answer(self):
        k = some_function(4)
        assert k == 5

    def test_dtype(self):
        k = some_function(4)
        assert type(k) == int