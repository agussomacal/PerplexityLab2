import inspect
import itertools
import multiprocessing
import os.path
from collections import OrderedDict, defaultdict, namedtuple
from contextlib import contextmanager
from pathos.multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Dict, Union, Set

import joblib
import numpy as np
import matplotlib.pyplot as plt

from perplexitylab.miscellaneous import make_hash


# ---------- Parallel or not parallel ----------
def get_workers(workers):
    if workers > 0:
        return min((multiprocessing.cpu_count() - 1, workers))
    else:
        return max((1, multiprocessing.cpu_count() + workers))


def get_appropiate_number_of_workers(workers, n):
    return int(np.max((1, np.min((multiprocessing.cpu_count() - 1, n, workers)))))


def get_map_function(workers=1):
    return map if workers == 1 else Pool(get_workers(workers)).imap_unordered


# ---------- Dict tools ----------
def filter_dict(dictionary: Dict, keys: Union[Set, List] = (), keys_not: Union[Set, List] = ()):
    assert set(keys).intersection(keys_not) == set(), \
        f"Keys and Keys_not should be disjoint but found common elements: {set(keys).intersection(keys_not)}"
    return {k: dictionary[k] for k in keys if k in dictionary.keys() and k not in keys_not}


# ---------- Verbosity tools ----------
@contextmanager
def message(msg_before, msg_after):
    print("\r" + msg_before, end='')
    yield
    print("\r" + msg_after)


# ================================================ #
#                ExperimentManager                 #
# ================================================ #
Task = namedtuple("Task", ["name", "task", "required_inputs"])

PROBLEM_LOADING = (None, None)

class ExperimentManager:
    def __init__(self, name, path, save_results=True, verbose=True, num_cpus=1, recalculate=False):
        self.name = name
        self.path = Path(path).joinpath(name)
        self.path.mkdir(parents=True, exist_ok=True)
        self.data_path = Path(path).joinpath(f".data_{name}")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.save_results = save_results
        self.verbose = verbose
        self.num_cpus = num_cpus
        self.recalculate = recalculate
        self.data_inputs_path = f"{self.data_path}/inputs.joblib"

        self.tasks = list()
        self.task_required_variables = OrderedDict()
        self.constants = OrderedDict()

    def set_constants(self, **constants):
        self.constants = constants

    def set_pipeline(self, **kwargs: Callable):
        for task_name, task in kwargs.items():
            self.tasks.append(Task(name=task_name, task=task, required_inputs=tuple(inspect.getfullargspec(task)[0])))

    def run_pipeline(self, **variables: List):
        _, explored_inputs = self.load_explored_inputs()
        result_dict = defaultdict(list)
        # inputs = self.constants.copy()
        variables.update({k: [v] for k, v in self.constants.items()})
        pmap = get_map_function(self.num_cpus)
        for i, results in enumerate(pmap(lambda val: list(self.run_tasks(dict(zip(variables, val)))),
                                         itertools.product(*variables.values()))):
            for singe_result in results:
                new_inputs = [singe_result[k] for k in variables.keys()]
                if self.verbose: print("_____________________")
                if self.verbose: print("(", i, ") Variables:", dict(zip(variables, new_inputs)))
                for k, v in singe_result.items():
                    result_dict[k].append(v)

                for saved_inputs in explored_inputs:
                    if all([v1 == v2 for v1, v2 in zip(new_inputs, saved_inputs)]): break
                else:
                    explored_inputs.append(new_inputs)
        self.save_explored_inputs(input_names=tuple(list(variables.keys())), explored_inputs=explored_inputs)
        return result_dict

    def run_tasks(self, inputs: Dict, order=0):
        if order >= len(self.tasks):
            yield inputs
        else:
            task_name, task, required_inputs = self.tasks[order]
            inputs_for_task = filter_dict(inputs, required_inputs)
            # ----- Load Execute Save ----- #
            single_value_variables, multiple_value_variables = PROBLEM_LOADING
            stored_result_filepath = self.get_stored_result_filepath(task_name, task, inputs_for_task)
            if os.path.exists(stored_result_filepath) and not self.recalculate:
                single_value_variables, multiple_value_variables = self.load_single_task_result(stored_result_filepath)
            # In case there was a problem loading recalculates
            if (single_value_variables, multiple_value_variables) == PROBLEM_LOADING:
                result = task(**inputs_for_task)
                # TODO: this might break if user puts a tuple output
                single_value_variables, multiple_value_variables = result if isinstance(result, tuple) and len(result) == 2 else (result, dict())
                if self.save_results:
                    self.save_single_task_result(single_value_variables, multiple_value_variables,
                                                 stored_result_filepath)
            # ----- Run next nested task ----- #
            inputs.update(single_value_variables)
            for new_variable_values in itertools.product(*multiple_value_variables.values()):
                inputs.update(dict(zip(multiple_value_variables, new_variable_values)))
                for single_result in self.run_tasks(inputs, order=order + 1):
                    yield single_result

    def get_stored_result_filepath(self, task_name, task, inputs_for_task):
        h = make_hash((task_name, task, inputs_for_task))
        return f"{self.data_path}/result_{h}.joblib"

    def save_single_task_result(self, single_value_variables, multiple_value_variables, stored_result_filepath):
        with message("Experiment finished -> saving...", "Experiment finished -> saved."):
            joblib.dump((single_value_variables, multiple_value_variables), stored_result_filepath)

    def load_single_task_result(self, stored_result_filepath):
        with message("Experiment alredy done -> loading...", "Experiment alredy done -> loaded."):
            try:
                return joblib.load(stored_result_filepath)
            except EOFError:
                return PROBLEM_LOADING

    def save_explored_inputs(self, input_names, explored_inputs):
        return joblib.dump((input_names, explored_inputs), self.data_inputs_path)

    def load_explored_inputs(self):
        return joblib.load(self.data_inputs_path) if os.path.exists(self.data_inputs_path) else ([], [])

    def load_results(self):
        input_names, explored_inputs = self.load_explored_inputs()
        result_dict = defaultdict(list)
        inputs = self.constants.copy()
        for single_experiment_variables in explored_inputs:
            inputs.update(dict(zip(input_names, single_experiment_variables)))
            for singe_result in self.run_tasks(inputs=inputs):
                for k, v in singe_result.items():
                    result_dict[k].append(v)
        return result_dict


# ================================================ #
#                     Plotting                     #
# ================================================ #
@contextmanager
def savefigure(path2plot, dpi=None):
    Path(path2plot).parent.mkdir(parents=True, exist_ok=True)
    yield
    plt.savefig(path2plot, dpi=dpi)
    plt.close()


def plottify(plot_func):
    def wrapper(em: ExperimentManager, filename: str, folder="", path=None, *args, **kwargs):
        variables = set(itertools.chain(*[task.required_inputs for task in em.tasks]))
        results = em.run_pipeline(**filter_dict(kwargs, variables))
        results = filter_dict(results, inspect.getfullargspec(plot_func)[0])  # get only variables relevant for plot
        kwargs = filter_dict(kwargs, inspect.getfullargspec(plot_func)[0])  # get only params relevant for plot
        kwargs = filter_dict(kwargs, keys_not=list(results.keys()))  # get only params for plot (not already in results)
        path2figure = f"{path if path is not None else em.path}/{folder}/{filename}"
        with savefigure(path2figure):
            plot_func(*args, **kwargs, **results)
        return path2figure

    return wrapper
