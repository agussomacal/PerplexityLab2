import os
import shutil
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from perplexitylab.experiment_tools import ExperimentManager, Task
from perplexitylab.plot_tools import plottify, plx_lineplot


class TestPipelines(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).parent.joinpath(".TestExperiments")
        self.path.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.em = ExperimentManager(name="Experiment", path=self.path, save_results=True)
        self.em.set_defaults(b=0.5)
        self.em.set_pipeline(
            Task(lambda a, x, b: {"out1": a * x ** 2 + b}),
            Task(lambda out1, x: {"out2": [out1] * 10, "out3": [x] * 10}),
        )

    def test_pipeline_plot_results(self):
        @plottify()
        def plot(a, x, out1):
            pd.DataFrame({"a": a, "x": x, "out1": out1}).groupby("a").plot(x="x", y="out1")

        path2plot = plot(
            em=self.em,
            filename="test_plot.png",
            x=np.linspace(-1, 1, 10),
            a=[1, 2],
        )
        assert os.path.exists(path2plot)

        @plottify(variables_assumed_unique=("a",))
        def plot(a, x, out1):
            (np.array(out1) - np.array(x)) * a

        self.em.set_defaults(a=1)
        path2plot = plot(
            em=self.em,
            filename="test_plot.png",
            x=np.linspace(-1, 1, 10),
        )

    def test_plx_lineplot(self):
        plx_lineplot(x_var="x", y_var="a", label_var="b")(
            em=self.em,
            filename="test_plot.png",
            x=np.linspace(-1, 1, 10),
            a=[1, 2],
            log="x",
        )

    def test_plx_lineplot_unfold(self):
        plx_lineplot(x_var="out3", y_var="out2", label_var="b")(
            em=self.em,
            filename="test_plot.png",
            x=np.linspace(-1, 1, 10),
            a=[1, 2],
            log="x",
        )

    def tearDown(self):
        shutil.rmtree(self.path)

    if __name__ == '__main__':
        unittest.main()
