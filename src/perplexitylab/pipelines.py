import inspect
import itertools
import os.path
from collections import OrderedDict, defaultdict, namedtuple
from pathlib import Path
from typing import Callable, List, Dict

import joblib

from src.perplexitylab.miscellaneous import make_hash


def filter_dict(keys, dictionary):
    return {k: dictionary[k] for k in keys if k in dictionary}


Task = namedtuple("Task", ["name", "task", "required_inputs"])


class ExperimentManager:
    def __init__(self, name, path, save_results=True):
        self.name = name
        self.path = path
        self.data_path = Path(path).joinpath(f".data_{name}")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.save_results = save_results
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
        inputs = self.constants.copy()
        for single_experiment_variables in itertools.product(*variables.values()):
            inputs.update(dict(zip(variables, single_experiment_variables)))
            for singe_result in self.run_tasks(inputs=inputs):
                for k, v in singe_result.items():
                    result_dict[k].append(v)

                new_inputs = [singe_result[k] for k in variables.keys()]
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
            inputs_for_task = filter_dict(required_inputs, inputs)
            # ----- Load Execute Save ----- #
            stored_result_filepath = self.get_stored_result_filepath(task_name, task, inputs_for_task)
            if os.path.exists(stored_result_filepath):
                single_value_variables, multiple_value_variables = self.load_single_task_result(stored_result_filepath)
            else:
                single_value_variables, multiple_value_variables = task(**inputs_for_task)
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
        joblib.dump((single_value_variables, multiple_value_variables), stored_result_filepath)

    def load_single_task_result(self, stored_result_filepath):
        return joblib.load(stored_result_filepath)

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
