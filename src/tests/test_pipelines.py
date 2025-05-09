import shutil
import time
import unittest
from pathlib import Path

from src.perplexitylab.pipelines import ExperimentManager


class TestVizUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).parent.joinpath(".TestExperiments")

    def test_pipeline(self):
        em = ExperimentManager(name="Experiment", path=self.path, save_results=False)
        em.set_constants(a=2)
        em.set_pipeline(
            function1=lambda a, x: ({"out1": a * x}, dict()),
            function2=lambda x, y: ({"out2": x + y}, dict()),
        )
        results = em.run_pipeline(
            x=[1, 2, 3],
            y=[10, 20],
        )

        assert len(results) == 5
        assert set(map(len, results.values())) == {6}

    def test_pipeline_nested_variables(self):
        em = ExperimentManager(name="Experiment", path=self.path, save_results=False)
        em.set_constants(a=2)
        em.set_pipeline(
            function1=lambda a, x: ({"out1": a * x}, {"out_multi": [a * x, a * x + 1]}),
            function2=lambda a, x, y, out_multi: ({"out2": out_multi - a * x + y}, dict()),
        )
        results = em.run_pipeline(
            x=[1, 2, 3],
            y=[10, 20],
        )

        assert len(results) == 6
        assert set(map(len, results.values())) == {12}

    def test_pipeline_save_load(self):
        em = ExperimentManager(name="Experiment", path=self.path, save_results=True)
        em.set_constants(a=2)
        em.set_pipeline(
            function1=lambda a, x: ({"out1": a * x}, {"out_multi": [a * x, a * x + 1]}),
            function2=lambda a, x, y, out_multi: ({"out2": out_multi - a * x + y}, dict()),
            function3=lambda a, x, y, out_multi: ({"out2": time.sleep(0.1)}, dict()),
        )
        t0 = time.time()
        em.run_pipeline(
            x=[1, 2, 3],
            y=[10, 20],
        )
        t = time.time() - t0
        assert t > 1

        t0 = time.time()
        em.run_pipeline(
            x=[1, 2, 3],
            y=[10, 20],
        )
        t = time.time() - t0
        assert t < 1

    def test_pipeline_load_results(self):
        em = ExperimentManager(name="Experiment", path=self.path, save_results=True)
        em.set_constants(a=2)
        em.set_pipeline(
            function1=lambda a, x: ({"out1": a * x}, {"out_multi": [a * x, a * x + 1]}),
            function2=lambda a, x, y, out_multi: ({"out2": out_multi - a * x + y}, dict()),
            function3=lambda a, x, y, out_multi: ({"out2": time.sleep(0.1)}, dict()),
        )
        results = em.run_pipeline(
            x=[1, 2, 3],
            y=[10, 20],
        )

        results_loaded = em.load_results()
        assert results == results_loaded

    def tearDown(self):
        shutil.rmtree(self.path)

    if __name__ == '__main__':
        unittest.main()
