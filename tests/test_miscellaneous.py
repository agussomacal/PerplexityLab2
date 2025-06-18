import os.path
import shutil
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from perplexitylab.miscellaneous import if_exist_load_else_do, make_hash, plx_partial, plx_partial_class, \
    DictList, group


class TestVizUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).parent.joinpath(".TestMicellaneous")
        self.path.mkdir(parents=True, exist_ok=True)

    def test_group(self):
        d = {
            "a": [1, 2, 3],
            "b": [1, 1, 3],
            "c": [1, 2, 2],
            "d": ["1", "1", "1"],
        }
        assert len(list(group(d, "a"))) == 3
        assert len(list(group(d, "b"))) == 2
        assert len(list(group(d, "c"))) == 2
        assert len(list(group(d, "d"))) == 1

    def test_plx_partial(self):
        def add(a, b):
            return a + b

        padd = plx_partial(add, b=2)
        assert hasattr(padd, '__name__')
        assert padd != add
        assert padd.__name__ != add.__name__
        assert add.__name__ in padd.__name__

    def test_plx_partial_class(self):
        @dataclass
        class Circle:
            radius: float = 1.

        Circle2 = plx_partial_class(Circle, radius=2.)
        assert Circle2().radius == 2.
        assert Circle2().__class__.__name__ == "plx_" + Circle.__name__

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

    def tearDown(self):
        shutil.rmtree(self.path)

    if __name__ == '__main__':
        unittest.main()


class TestDictLike(unittest.TestCase):

    def test_update(self):
        d = DictList()
        self.assertEqual(d.data, {})

        d.update({'a': [1, 2], 'b': [3, 4]})
        self.assertEqual(d.data, {'a': [1, 2], 'b': [3, 4]})

    def test_update_merges_lists(self):
        d = DictList()
        d.update({'a': [1, 2]})
        d.update({'a': [3, 4]})
        self.assertEqual(d.data, {'a': [1, 2, 3, 4]})

    def test_update_adds_new_keys(self):
        d = DictList()
        d.update({'a': [1, 2]})
        d.update({'b': [3, 4]})
        self.assertEqual(d.data, {'a': [1, 2], 'b': [3, 4]})

    def test_getitem(self):
        d = DictList()
        d['a'] = [1, 2]
        self.assertEqual(d['a'], [1, 2])

    def test_setitem(self):
        d = DictList()
        d['a'] = [1, 2]
        self.assertEqual(d.data, {'a': [1, 2]})

    def test_todict(self):
        d = DictList()
        d['a'] = [1, 2]
        self.assertEqual(d.todict(), {"a": [1, 2]})

    if __name__ == '__main__':
        unittest.main()
