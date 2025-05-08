import os.path
import unittest
from pathlib import Path

import numpy as np

from src.perplexitylab.miscellaneous import if_exist_load_else_do, make_hash


class Foo1:
    def __init__(self, a):
        self.a = a


class Foo2(Foo1):
    def __init__(self, a, b):
        super().__init__(a=a)
        self.b = b


class TestVizUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).parent.joinpath("TestMicellaneous")

    def test_hash(self):
        assert make_hash(1) == make_hash(2 - 1)
        assert make_hash((1, 2)) == make_hash((1, 2))
        assert make_hash((1, 2)) != make_hash((2, 1))
        assert make_hash({"a": 2, "b": 5}) == make_hash({"b": 5, "a": 2})
        v = np.random.uniform(0, 1, size=100)
        assert make_hash({"a": v, "b": 5}) == make_hash({"b": 5, "a": v})
        k = np.random.uniform(0, 1, size=100)
        assert make_hash(dict(zip(k, v))) == make_hash(dict(zip(k, v)))

    def test_if_exist_load_else_do(self):
        @if_exist_load_else_do(file_format="joblib", loader=None, saver=None,
                               description=lambda data: print(data), check_hash=False)
        def do_something(a, b):
            return a + b

        path2file = f"{self.path}/do_something.joblib"
        if os.path.exists(path2file):
            os.remove(path2file)

        path2hash = f"{self.path}/do_something.hash"
        if os.path.exists(path2hash):
            os.remove(path2hash)

        res = do_something(path=self.path, filename=None, recalculate=False, a=1, b=2)
        assert os.path.exists(path2file)
        assert not os.path.exists(path2hash)
        assert res == 3
        res = do_something(path=self.path, filename=None, recalculate=False, a=1, b=2)
        assert res == 3

        # change input but check_hash = False
        res = do_something(path=self.path, filename=None, recalculate=False, a=1, b=5)
        assert res == 3

    def test_if_exist_load_else_do_hash(self):
        @if_exist_load_else_do(file_format="joblib", loader=None, saver=None,
                               description=lambda data: print(data), check_hash=True)
        def do_something(a, b):
            return a + b

        path2file = f"{self.path}/do_something.joblib"
        if os.path.exists(path2file):
            os.remove(path2file)

        path2hash = f"{self.path}/do_something.hash"
        if os.path.exists(path2hash):
            os.remove(path2hash)

        res = do_something(path=self.path, filename=None, recalculate=False, a=1, b=2)
        assert os.path.exists(path2file)
        assert os.path.exists(path2hash)
        assert res == 3
        res = do_something(path=self.path, filename=None, recalculate=False, a=1, b=2)
        assert res == 3

        # change input but check_hash = False
        res = do_something(path=self.path, filename=None, recalculate=False, a=1, b=5)
        assert res == 6

    if __name__ == '__main__':
        unittest.main()
