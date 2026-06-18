import functools
import hashlib
import inspect
import multiprocessing
import os
import pickle
import shutil
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Type, Union, Set, List, Optional, Any, Tuple

import joblib
import numpy as np
from matplotlib import pyplot as plt
from pathos.multiprocessing import Pool


# ---------- Parallel or not parallel ----------
def get_workers(workers):
    if workers > 0:
        return min((multiprocessing.cpu_count() - 1, workers))
    else:
        return max((1, multiprocessing.cpu_count() + workers))


def get_appropriate_number_of_workers(workers, n):
    return int(np.max((1, np.min((multiprocessing.cpu_count() - 1, n, workers)))))


def get_map_function(workers=1):
    return map if workers == 1 else Pool(get_workers(workers)).imap_unordered


# ---------- Dict tools ----------
def filter_dict(dictionary: Dict, keys: Union[Set, List] = None, keys_not: Union[Set, List] = ()):
    keys = dictionary.keys() if keys is None else keys
    return {k: dictionary[k] for k in keys if k in dictionary.keys() and k not in keys_not}


def filter_for_func(func: Callable, dictionary: Dict):
    return filter_dict(dictionary, inspect.getfullargspec(func)[0])


class DictList:
    def __init__(self):
        self.data = dict()

    def append(self, d):
        self.update({k: [v] for k, v in d.items()})

    def update(self, d):
        for key, value in d.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].extend(value)

    def __getitem__(self, key):
        return self.data.get(key, [])

    def __setitem__(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].extend(value)

    def todict(self):
        return self.data


def group(dictionary, *keys):
    if len(keys) == 0:
        yield dictionary.copy(), dict()
    else:
        membership = np.array(list(map(make_hash, zip(*[dictionary[k] for k in keys]))))
        classes = np.unique(membership)
        for c in classes:
            yield ({k: [e for i, e in enumerate(v) if membership[i] == c] for k, v in dictionary.items()},
                   {k: [e for i, e in enumerate(dictionary[k]) if membership[i] == c][0] for k in keys})


# ---------- Verbosity tools ----------
@contextmanager
def message(msg_before, msg_after):
    print("\r" + msg_before, end='')
    yield
    print("\r" + msg_after)


# ----------- time -----------
@contextmanager
def timeit(msg, verbose=True):
    if verbose:
        print(msg, end='')
    t0 = time.time()
    yield
    if verbose:
        print('\r -> duracion {}: {:.2f}s'.format(msg, time.time() - t0))


# ----------- plots -----------
def set_latex_fonts(font_family="amssymb", packages=("amsmath",)):
    # ("babel", "lmodern", "amsmath", "amsthm", "amssymb", "amsfonts", "fontenc", "inputenc")
    # preamble=r'\usepackage{babel}\usepackage{lmodern}\usepackage{amsmath,amsthm,amssymb}\usepackage{amsfonts}\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}'
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": font_family,
    })
    plt.rc('text.latex',
           preamble=r''.join([f"\\usepackage{{{package}}}" for package in packages])
           # )
           )


# ----------- func utils -----------
def plx_partial(function: Callable, prefix="plx_partial_", suffix="", function_rename=None,
                use_kwargs_for_as_suffix=False, *args, **kwargs) -> Callable:
    partial_function = functools.partial(function, *args, **kwargs)
    if use_kwargs_for_as_suffix: suffix += " " + " ".join([f"{k}: {v}" for k, v in kwargs.items()])
    partial_function.__name__ = prefix + (function.__name__ if function_rename is None else function_rename) + suffix
    # partial_function.__module__ = "plx_partial_" + function.__module__
    return partial_function


def plx_partial_class(class_type: Type, prefix="plx_", suffix="", *args, **kwargs) -> Type:
    return type(prefix + class_type.__name__ + suffix, (class_type,),
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
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace(";", "").replace(
        ":", "").replace(" ", "_")
    s = s[0] + s[1:].replace(".", "")  # except if it is a hidden file.
    return s


DictProxyType = type(object.__dict__)


def make_hash(o: Any) -> str:
    """
    Recursively makes a hashable representation of complex objects (dicts, lists, sets, arrays).
    Returns a hexadecimal SHA256 hash string.
    """
    # Handle None explicitly
    if o is None:
        return hashlib.sha256(b"None").hexdigest()

    # Strings: Encode to UTF-8 and hash
    if isinstance(o, str):
        return hashlib.sha256(o.encode('utf-8')).hexdigest()

    # Sets: Sort elements to ensure order independence
    if isinstance(o, set):
        # Sort requires elements to be comparable; fallback to string repr if not
        try:
            sorted_o = sorted(list(o))
        except TypeError:
            sorted_o = sorted([str(x) for x in o])
        return make_hash(sorted_o)

    # Lists/Tuples: Process recursively
    if isinstance(o, (list, tuple)):
        # Create a tuple of hashes to maintain structure
        hashed_items = tuple(make_hash(item) for item in o)
        # Hash the resulting tuple string representation to get a single hash
        return hashlib.sha256(str(hashed_items).encode('utf-8')).hexdigest()

    # Dictionaries: Sort keys to ensure order independence
    if isinstance(o, dict):
        sorted_items = tuple((k, v) for k, v in sorted(o.items()))
        hashed_items = tuple((make_hash(k), make_hash(v)) for k, v in sorted_items)
        return hashlib.sha256(str(hashed_items).encode('utf-8')).hexdigest()

    # Numpy Arrays: Convert to list (efficient for large arrays? maybe use tobytes())
    if isinstance(o, np.ndarray):
        # Using .tobytes() is faster and more memory efficient for large arrays than .tolist()
        return hashlib.sha256(o.tobytes()).hexdigest()

    # Classes: Hash name (and optionally module)
    if inspect.isclass(o):
        return hashlib.sha256(f"{o.__module__}.{o.__name__}".encode('utf-8')).hexdigest()

    # Functions: Hash name (Code object hashing can be fragile across environments)
    if callable(o) and not inspect.isclass(o):
        # Warning: __code__.co_filename changes between dev/prod.
        # Hashing just the name is safer for portability.
        return hashlib.sha256(o.__name__.encode('utf-8')).hexdigest()

    # Numbers (int, float)
    if isinstance(o, (int, float)):
        # Ensure consistent string representation
        return make_hash(str(o))

    # Fallback: Convert everything else to string
    return hashlib.sha256(str(o).encode('utf-8')).hexdigest()


def ifex_saver(data: Any, filepath: str, saver: Optional[Callable] = None, file_format: str = "joblib",
               verbose: bool = True):
    """
    Saves data to a file using the specified format or custom saver function.
    CRITICAL: Uses binary modes ('wb') for pickle to prevent corruption.
    """
    # Use context manager if available, otherwise just print timing
    # Assuming 'timeit' is a context manager defined elsewhere
    timer_ctx = timeit(f"Saving processed {filepath}:", verbose=verbose)

    with timer_ctx:
        if saver is not None:
            saver(data, filepath)
            return

        format_lower = file_format.lower().strip()

        if format_lower == "npy":
            np.save(filepath, data)

        elif format_lower == "pickle":
            # FIX: Use 'wb' (write binary) instead of 'r'
            with open(filepath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif format_lower == "joblib":
            joblib.dump(data, filepath)

        else:
            raise ValueError(f"Format '{file_format}' not implemented. Supported: npy, pickle, joblib.")


def ifex_loader(filepath: str, loader: Optional[Callable] = None, file_format: str = "joblib",
                verbose: bool = True) -> Any:
    """
    Loads data from a file using the specified format or custom loader function.
    CRITICAL: Uses binary modes ('rb') for pickle.
    """
    timer_ctx = timeit(f"Loading pre-processed {filepath}:", verbose=verbose)

    with timer_ctx:
        if loader is not None:
            return loader(filepath)

        format_lower = file_format.lower().strip()

        if format_lower == "npy":
            # allow_pickle=True allows loading of object arrays if necessary
            return np.load(filepath, allow_pickle=True)

        elif format_lower == "pickle":
            # FIX: Use 'rb' (read binary) instead of 'r'
            with open(filepath, "rb") as f:
                return pickle.load(f)

        elif format_lower == "joblib":
            return joblib.load(filepath)

        else:
            raise ValueError(f"Format '{file_format}' not implemented. Supported: npy, pickle, joblib.")


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
            if recalculate or not os.path.exists(filepath) or (check_hash and not_same_hash):
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


def check_do_save_or_load_experiment(default_path=None, embedd_in_folder=True, file_format="joblib", loader=None,
                                     saver=None, description=None, vars_filter=None):
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
    :vars_filter=None when it is not it should be a function
    """

    assert vars_filter is None or isinstance(vars_filter, Callable), \
        ("vars_filter should be a function that takes the input vriables of the function and returns a subset or "
         "transformation that will be used to create the hashh and save the experiment")

    def decorator(do_func):
        def decorated_func(path=None, filename=None, experiment_tag="", recalculate=False, save=True, verbose=True,
                           *args,
                           **kwargs):
            if path is None:
                path = default_path

            if path is None:
                warnings.warn("Missing path: experiments won't be saved.")
                return do_func(*args, **kwargs)
            else:
                path = Path(path)
                if embedd_in_folder:
                    path = path.joinpath(
                        embedd_in_folder if isinstance(embedd_in_folder, str) else f".{do_func.__name__}")

                path.mkdir(parents=True, exist_ok=True)
                hash_of_input = make_hash((experiment_tag,) +
                                          ((args, kwargs) if vars_filter is None else vars_filter(*args, **kwargs)))
                filename = f".{do_func.__name__ if filename is None else filename}_{hash_of_input}"
                filename = clean_str4saving(filename)
                filepath = f"{path}/{filename}.{file_format}"

                # process save or load
                if not save or recalculate or not os.path.exists(filepath):
                    # Processing
                    with timeit(f"Processing {filepath}:", verbose=verbose):
                        data = do_func(*args, **kwargs)

                    # Saving data and hash
                    if save: ifex_saver(data, filepath=filepath, saver=saver, file_format=file_format)
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
