import copy
import inspect
import itertools
import os.path
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Dict

import dill
import joblib

from perplexitylab.miscellaneous import make_hash, get_map_function, filter_dict, message


# ================================================ #
#                ExperimentManager                 #
# ================================================ #
@dataclass
class Task:
    function: Callable
    save: bool = True
    task_name: str = None

    @property
    def required_inputs(self):
        return tuple(inspect.getfullargspec(self.function)[0])

    @property
    def name(self):
        return self.function.__name__ if self.task_name is None else self.task_name


PROBLEM_LOADING = (None, None)
PROBLEM_LOADING_EXPLORED_INPUTS = ([], [])


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
        self.data_inputs_path_readable = f"{self.data_path}/inputs.txt"

        self.tasks = list()
        self.task_required_variables = OrderedDict()
        self.constants = OrderedDict()

    def set_defaults(self, **constants):
        self.constants.update(**constants)

    def set_pipeline(self, *tasks: Task):
        self.tasks = tasks

    def add_post_processing(self, *tasks: Task):
        new_experiment_manager = copy.deepcopy(self)
        new_experiment_manager.set_pipeline(*self.tasks, *tasks)
        return new_experiment_manager

    def run_pipeline(self, **kwargs: List):
        _, explored_inputs = self.load_explored_inputs()
        result_dict = defaultdict(list)
        variables = {k: [v] for k, v in self.constants.items()}
        variables.update(**kwargs)
        pmap = get_map_function(self.num_cpus)
        for i, results in enumerate(
                pmap(lambda val: list(self.run_tasks(dict(zip(variables, val)), input_hash=make_hash(val))),
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

    def run_tasks(self, inputs: Dict, input_hash, order=0):
        # TODO: h, the input_hash creates an output for al the graph even if some inputs are not used for an intermediate function
        if order >= len(self.tasks):
            yield inputs
        else:
            inputs_for_task = filter_dict(inputs, self.tasks[order].required_inputs)
            # ----- Load Execute Save ----- #
            single_value_variables, multiple_value_variables = PROBLEM_LOADING
            stored_result_filepath = self.get_stored_result_filepath(
                self.tasks[order].name, self.tasks[order].function, inputs_for_task, input_hash)
            if os.path.exists(stored_result_filepath) and not self.recalculate:
                single_value_variables, multiple_value_variables = self.load_single_task_result(stored_result_filepath)
            # In case there was a problem loading recalculates
            if (single_value_variables, multiple_value_variables) == PROBLEM_LOADING:
                result = self.tasks[order].function(**inputs_for_task)
                # TODO: this might break if user puts a tuple output
                single_value_variables, multiple_value_variables = result if isinstance(result, tuple) and len(
                    result) == 2 else (result, dict())
                if self.save_results and self.tasks[order].save:
                    self.save_single_task_result(single_value_variables, multiple_value_variables,
                                                 stored_result_filepath)
            # ----- Run next nested task ----- #
            inputs.update(single_value_variables)
            for new_variable_values in itertools.product(*multiple_value_variables.values()):
                new_inputs = {**inputs, **dict(zip(multiple_value_variables, new_variable_values))}
                for single_result in self.run_tasks(new_inputs, input_hash=input_hash, order=order + 1):
                    yield single_result

    def get_stored_result_filepath(self, task_name, task, inputs_for_task, h):
        h = make_hash((task_name, task, inputs_for_task, h))
        return f"{self.data_path}/result_{h}.joblib"

    def save_single_task_result(self, single_value_variables, multiple_value_variables, stored_result_filepath):
        with message("Experiment finished -> saving...", "Experiment finished -> saved."):
            joblib.dump((single_value_variables, multiple_value_variables), stored_result_filepath)

    def load_single_task_result(self, stored_result_filepath):
        with message("Experiment alredy done -> loading...", "Experiment alredy done -> loaded."):
            try:
                return joblib.load(stored_result_filepath)
            except EOFError:
                warnings.warn("Warning: some problem with stored RESULTS file, probably not correctly saved.")
        return PROBLEM_LOADING

    def save_explored_inputs(self, input_names, explored_inputs):
        # dill to pickle objects created on the fly
        dill.dump((input_names, explored_inputs), open(self.data_inputs_path, "wb"))

    def load_explored_inputs(self):
        if os.path.exists(self.data_inputs_path):
            try:
                # dill to pickle objects created on the fly
                return dill.load(open(self.data_inputs_path, "rb"))
            except EOFError:
                warnings.warn("Warning: some problem with explored INPUTS file, probably not correctly saved.")
        return PROBLEM_LOADING_EXPLORED_INPUTS

    def load_results(self):
        input_names, explored_inputs = self.load_explored_inputs()
        result_dict = defaultdict(list)
        inputs = self.constants.copy()
        for single_experiment_variables in explored_inputs:
            inputs.update(dict(zip(input_names, single_experiment_variables)))
            for singe_result in self.run_tasks(inputs=inputs, input_hash=make_hash(single_experiment_variables)):
                for k, v in singe_result.items():
                    result_dict[k].append(v)
        return result_dict
