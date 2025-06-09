import os
import shutil
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from perplexitylab.pipelines import ExperimentManager, plottify, Task


class TestPipelines(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).parent.joinpath(".TestExperiments")
        self.path.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

    def test_pipeline(self):
        em = ExperimentManager(name="Experiment", path=self.path, save_results=False)
        em.set_defaults(a=2)
        em.set_pipeline(
            Task(lambda a, x: {"out1": a * x}, task_name="function1"),
            Task(lambda x, y: {"out2": x + y}, task_name="function2"),
        )
        results = em.run_pipeline(
            x=[1, 2, 3],
            y=[10, 20],
        )

        assert len(results) == 5
        assert set(map(len, results.values())) == {6}

    def test_pipeline_parallel(self):
        em = ExperimentManager(name="Experiment", path=self.path, save_results=False, num_cpus=2)
        em.set_defaults(a=2)
        em.set_pipeline(
            Task(lambda a, x: {"out1": a * x}, task_name="function1"),
            Task(lambda x, y: {"out2": x + y}, task_name="function2"),
        )
        results = em.run_pipeline(
            x=[1, 2, 3],
            y=[10, 20],
        )

        assert len(results) == 5
        assert set(map(len, results.values())) == {6}

    def test_pipeline_nested_variables(self):
        em = ExperimentManager(name="Experiment", path=self.path, save_results=False)
        em.set_pipeline(
            Task(lambda x: ({"out1": x}, {"out_multi": [x, x + 1]}), task_name="function1"),
            Task(lambda x, y, out_multi: {"out2": out_multi - x - y}, task_name="function2"),
        )
        results = em.run_pipeline(
            x=[1, 2, 3],
            y=[1, 2],
        )

        assert len(results) == 5
        assert set(map(len, results.values())) == {12}

    def test_pipeline_save_load(self):
        em = ExperimentManager(name="Experiment", path=self.path, save_results=True, recalculate=True)
        em.set_defaults(a=2)
        em.set_pipeline(
            Task(lambda a, x: ({"out1": a * x}, {"out_multi": [a * x, a * x + 1]})),
            Task(lambda a, x, y, out_multi: {"out2": out_multi - a * x + y}),
            Task(lambda a, x, y, out_multi: {"out3": time.sleep(0.01)}),
        )
        t0 = time.time()
        em.run_pipeline(
            x=[1, 2, 3, 4, 5],
            y=[10, 20],
        )
        t = time.time() - t0
        assert t > 0.1

        em.recalculate = False
        t0 = time.time()
        em.run_pipeline(
            x=[1, 2, 3, 4, 5],
            y=[10, 20],
        )
        t = time.time() - t0
        assert t < 0.1

    def test_pipeline_recalculate(self):
        em = ExperimentManager(name="Experiment", path=self.path, save_results=True, recalculate=True)
        em.set_defaults(a=2)
        em.set_pipeline(
            Task(lambda a, x: ({"out1": a * x}, {"out_multi": [a * x, a * x + 1]})),
            Task(lambda a, x, y, out_multi: {"out2": out_multi - a * x + y}),
            Task(lambda a, x, y, out_multi: {"out3": time.sleep(0.01)}),
        )
        t0 = time.time()
        em.run_pipeline(
            x=[1, 2, 3],
            y=[10, 20],
        )
        t = time.time() - t0
        assert t > 0.1

        t0 = time.time()
        em.run_pipeline(
            x=[1, 2, 3],
            y=[10, 20],
        )
        t = time.time() - t0
        assert t > 0.1

    def test_pipeline_load_results(self):
        em = ExperimentManager(name="Experiment", path=self.path, save_results=True)
        em.set_defaults(a=2)
        em.set_pipeline(
            Task(lambda a, x: {"out1": a * x}),
            Task(lambda a, x, y, out1: {"out2": out1 - a * x + y}),
            Task(lambda a, x, y, out1: {"out3": out1 + 2}),
        )
        results = em.run_pipeline(
            x=[1, 2, 3],
            y=[10, 20],
        )

        results_loaded = em.load_results()
        assert set(results.keys()) == set(results_loaded.keys())
        assert all([set(results[k]) == set(v) for k, v in results_loaded.items()])

    def test_pipeline_recalculating_tasks(self):
        em = ExperimentManager(name="Experiment", path=self.path, save_results=True)
        em.set_defaults(a=2)
        em.set_pipeline(
            Task(lambda a, x: {"out1": a * x}, save=False),
            Task(lambda y, out1: {"out3": out1 + y}),
        )
        results = em.run_pipeline(
            x=[1, 2, 3],
            y=[10, 20],
        )
        assert len(results["out3"]) == len([name for name in os.listdir(em.data_path)]) - 1
        em.set_pipeline(
            Task(lambda a, x: {"out1": a * x}, save=True),
            Task(lambda y, out1: {"out3": out1 + y}),
        )
        results = em.run_pipeline(
            x=[1, 2, 3],
            y=[10, 20],
        )
        # assert len(results["out3"]) + len(set(results["x"])) == len([name for name in os.listdir(em.data_path)]) - 1
        assert len(results["out3"]) * 2 == len([name for name in os.listdir(em.data_path)]) - 1

    def test_pipeline_plot_results(self):
        em = ExperimentManager(name="Experiment", path=self.path, save_results=True)
        em.set_defaults(b=0.5)
        em.set_pipeline(
            Task(lambda a, x, b: ({"out1": a * x ** 2 + b}, dict())),
        )

        @plottify()
        def plot(a, x, out1):
            pd.DataFrame({"a": a, "x": x, "out1": out1}).groupby("a").plot(x="x", y="out1")

        path2plot = plot(
            em=em,
            filename="test_plot.png",
            x=np.linspace(-1, 1, 10),
            a=[1, 2],
        )
        assert os.path.exists(path2plot)

        @plottify(variables_assumed_unique=("a",))
        def plot(a, x, out1):
            (np.array(out1) - np.array(x)) * a

        em.set_defaults(a=1)
        path2plot = plot(
            em=em,
            filename="test_plot.png",
            x=np.linspace(-1, 1, 10),
        )

    def tearDown(self):
        shutil.rmtree(self.path)

    if __name__ == '__main__':
        unittest.main()
