import functools
import inspect
import os
import pickle
import shutil
import time
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial, partialmethod
from pathlib import Path
from typing import Callable, Dict, Type
import hashlib

import joblib
import numpy as np


# ----------- time -----------
@contextmanager
def timeit(msg, verbose=True):
    if verbose:
        print(msg, end='')
    t0 = time.time()
    yield
    if verbose:
        print('\r -> duracion {}: {:.2f}s'.format(msg, time.time() - t0))


# ----------- func utils -----------
def plx_partial(function: Callable, *args, **kwargs) -> Callable:
    partial_function = functools.partial(function, *args, **kwargs)
    partial_function.__name__ = "plx_partial_" + function.__name__
    # partial_function.__module__ = "plx_partial_" + function.__module__
    return partial_function


def plx_partial_class(class_type: Type, *args, **kwargs) -> Type:
    return type("plx_" + class_type.__name__, (class_type,),
                {"__init__": lambda self, *arg2, **kwargs2: plx_partial(class_type.__init__, *args, **kwargs)(self=self,
                                                                                                              *arg2,
                                                                                                              **kwargs2)})


# ---------- File utils ---------- #
def copy_main_script_version(file, results_path):
    shutil.copyfile(os.path.realpath(file), f"{results_path}/main_script.py")


def check_create_path(path, *args):
    path = Path(path)
    for name in args:
        path = path.joinpath(name)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def clean_str4saving(s):
    return s.replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace(
        ".", "").replace(";", "").replace(":", "").replace(" ", "_")


import copy

DictProxyType = type(object.__dict__)


def make_hash(o):
    """
    Thanks to: https://stackoverflow.com/questions/5884066/hashing-a-dictionary

    Makes a hash from a dictionary, list, tuple or set to any level, that
    contains only other hashable types (including any lists, tuples, sets, and
    dictionaries). In the case where other kinds of objects (like classes) need
    to be hashed, pass in a collection of object attributes that are pertinent.
    For example, a class can be hashed in this fashion:

      make_hash([cls.__dict__, cls.__name__])

    A function can be hashed like so:

      make_hash([fn.__dict__, fn.__code__])
    """

    # if type(o) == DictProxyType:
    #     o2 = {}
    #     for k, v in o.items():
    #         if not k.startswith("__"):
    #             o2[k] = v
    #     o = o2

    if isinstance(o, str):
        return hashlib.sha256(o.encode('utf-8')).hexdigest()
    elif isinstance(o, (set, tuple, list)):
        return make_hash(str([make_hash(e) for e in o]))
    elif isinstance(o, set):
        return make_hash(sorted(list(o)))
    elif isinstance(o, dict):
        return make_hash(tuple([(make_hash(k), make_hash(o[k])) for k in sorted(o)]))
    elif isinstance(o, np.ndarray):
        return make_hash(o.ravel().tolist())
    elif inspect.isclass(o):
        return make_hash([o.__name__, ])  # TODO: check the right way to hash
    elif isinstance(o, Callable):
        return make_hash([o.__name__, o.__dict__, ])  # TODO: check the right way to hash/ , o.__code__.co_filename
    elif isinstance(o, (int, float)):
        return make_hash(str(o))
    else:
        return make_hash(str(o))
    # new_o = copy.deepcopy(o)
    # for k, v in new_o.items():
    #     new_o[k] = make_hash(v)
    #
    # return hash(tuple(frozenset(sorted(new_o.items()))))


def ifex_saver(data, filepath, saver, file_format):
    with timeit(f"Saving processed {filepath}:"):
        if saver is not None:
            saver(data, filepath)
        elif "npy" in file_format:
            np.save(filepath, data)
        elif "pickle" in file_format:
            with open(filepath, "r") as f:
                pickle.dump(data, f)
        elif "joblib" in file_format:
            joblib.dump(data, filepath)
        else:
            raise Exception(f"Format {file_format} not implemented.")


def ifex_loader(filepath, loader, file_format):
    """
    :param file_format: format for the termination of the file. If not known specify loader an saver. known ones are: npy, pickle, joblib
    :param loader: function that knows how to load the file
    """
    with timeit(f"Loading pre-processed {filepath}:"):
        if loader is not None:
            data = loader(filepath)
        elif "npy" == file_format:
            data = np.load(filepath, allow_pickle=True)
        elif "pickle" == file_format:
            with open(filepath, "r") as f:
                data = pickle.load(f)
        elif "joblib" == file_format:
            data = joblib.load(filepath)
        else:
            raise Exception(f"Format {file_format} not implemented.")
        return data


def if_exist_load_else_do(file_format="joblib", loader=None, saver=None, description=None, check_hash=False):
    """
    Decorator to manage loading and saving of files after a first processing execution.
    :param file_format: format for the termination of the file. If not known specify loader an saver. known ones are: npy, pickle, joblib
    :param loader: function that knows how to load the file
    :param saver: function that knows how to save the file
    :param description: description of data as a function depending on the type of data.
    :return:
    a function with the same aprameters as the decorated plus
    :path to specify path to folder
    :filename=None to specify filename
    :recalculate=False to force recomputation.
    :check_hash=False to recalculate if inputs to function change with respect of saved data
    :save=True to save
    """

    def decorator(do_func):
        def decorated_func(path, filename=None, recalculate=False, save=True, verbose=True, *args, **kwargs):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            filename = do_func.__name__ if filename is None else filename
            # if args4name == "all":
            #     args4name = sorted(kwargs.keys())
            # filename = f"{filename}" + "_".join(f"{arg_name}{kwargs[arg_name]}" for arg_name in args4name)
            filename = clean_str4saving(filename)
            filepath = f"{path}/{filename}.{file_format}"
            filepath_hash = f"{path}/{filename}.hash"

            # get hash of old and new file
            if check_hash:
                hash_of_input_old = None
                if os.path.exists(filepath_hash):
                    with open(filepath_hash, "r") as f:
                        hash_of_input_old = f.readline()
                hash_of_input = make_hash((args, kwargs))
                not_same_hash = (hash_of_input != hash_of_input_old)
            else:
                not_same_hash = True

            # process save or load
            if not save or recalculate or not os.path.exists(filepath) or (check_hash and not_same_hash):
                # Processing
                with timeit(f"Processing {filepath}:", verbose=verbose):
                    data = do_func(*args, **kwargs)

                # Saving data and hash
                if save:
                    ifex_saver(data, filepath=filepath, saver=saver, file_format=file_format)
                if check_hash:
                    with open(filepath_hash, "w") as f:
                        f.writelines(str(hash_of_input))
            else:
                # loading data
                try:
                    data = ifex_loader(filepath=filepath, loader=loader, file_format=file_format)
                except EOFError:
                    raise Exception(f"Problem with the file: {filepath}. Try deleting it to recalculate it.")

            # do post processing
            if isinstance(description, Callable):
                description(data)
            return data

        return decorated_func

    return decorator
