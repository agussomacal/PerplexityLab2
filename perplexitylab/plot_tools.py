import inspect
import itertools
from contextlib import contextmanager
from email.policy import default
from pathlib import Path
from typing import Callable, Iterable, Tuple, Dict, Any, List, Union

import pandas as pd
from fontTools.merge.util import equal
from makefun import with_signature
from matplotlib import pyplot as plt
import seaborn as sns

from perplexitylab.miscellaneous import filter_dict, filter_for_func, plx_partial
from perplexitylab.experiment_tools import ExperimentManager


# ================================================ #
#                    Plot tools                    #
# ================================================ #
@contextmanager
def savefigure(path2plot, dpi=None):
    Path(path2plot).parent.mkdir(parents=True, exist_ok=True)
    yield
    plt.savefig(path2plot, dpi=dpi)
    plt.close()


# ================================================ #
#          Connect plot with experiment            #
# ================================================ #
def plx_generic_plot_styler(figsize=(8, 6), log="", xlabel=None, ylabel=None, xlabel_fontsize=None,
                            ylabel_fontsize=None, legend_fontsize=None, legend_loc=None, bbox_to_anchor=None):
    def styler_function():
        fig, ax = plt.subplots(figsize=figsize)

        yield fig, ax

        if "x" in log: ax.set_xscale("log")
        if "y" in log: ax.set_yscale("log")
        if xlabel is not None: ax.set_xlabel(xlabel)
        if xlabel_fontsize is not None: ax.set_xlabel(ax.get_xlabel(), fontsize=xlabel_fontsize)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if ylabel_fontsize is not None: ax.set_ylabel(ax.get_ylabel(), fontsize=ylabel_fontsize)
        if legend_fontsize is not None or legend_loc is not None or bbox_to_anchor is not None:
            ax.legend(fontsize=legend_fontsize,
                      bbox_to_anchor=bbox_to_anchor,
                      loc=legend_loc)
        plt.tight_layout()

    return styler_function


def plottify(variables_assumed_unique=()):
    def decorator(plot_func):
        def wrapper(
                em: ExperimentManager, filename: str,
                experiment_setup: Union[
                    Tuple[Dict[str, Any], Dict[str, Any]], List[Tuple[Dict[str, Any], Dict[str, Any]]]],
                common_experiment_setup: Tuple[Dict[str, Any], Dict[str, Any]] = ({}, {}),
                folder="", path=None, verbose=True,
                style_function=plx_generic_plot_styler(), **kwargs):
            args4style = filter_for_func(style_function, kwargs)
            plot_func_variables = inspect.getfullargspec(plot_func).args
            assert "fig" in plot_func_variables, f"the input variable 'fig' should be in {plot_func.__name__} even if is unused."
            assert "ax" in plot_func_variables, f"the input variable 'ax' should be in {plot_func.__name__} even if is unused."
            # required_variables = set(plot_func_variables).intersection(variables)
            results = em.run_experiments(experiment_setup=experiment_setup, required_variables=plot_func_variables,
                                         common_experiment_setup=common_experiment_setup)
            results = filter_for_func(plot_func, results)  # get only variables relevant for plot
            kwargs = filter_for_func(plot_func, kwargs)  # get only params relevant for plot
            kwargs = filter_dict(kwargs, keys_not=list(results.keys()))  # get params for plot (not already in results)
            for k in variables_assumed_unique:
                results[k] = em.constants[k] if k in em.constants else results[k].pop()
            path2figure = f"{path if path is not None else em.path}/{folder}/{filename}"
            with savefigure(path2figure):
                with contextmanager(style_function)(**args4style) as (fig, ax):
                    plot_func(fig, ax, **kwargs, **results)
            if verbose: print("Figure saved in:", path2figure)
            return path2figure

        return wrapper

    return decorator


# ================================================ #
#                   Premade plots                  #
# ================================================ #
def unfold(x_var, y_var, label_var):
    unfolded_x = []
    unfolded_y = []
    unfolded_label = []
    for x, y, label in zip(x_var, y_var, label_var):
        are_iterables = list(map(lambda d: isinstance(d, Iterable) and not isinstance(d, str), (x, y, label)))
        if any(are_iterables):
            if not all(are_iterables):
                assert are_iterables[0] == are_iterables[1] == True, "x and y should be both iterables"
                assert len(x) == len(y), "x and y should have the same length"
                label = [label] * len(x)
            assert len(set(map(len, (x, y, label)))) == 1, "x, y and labels must have the same length"
            unfolded_x += list(x)
            unfolded_y += list(y)
            unfolded_label += list(label)
        else:
            unfolded_x.append(x)
            unfolded_y.append(y)
            unfolded_label.append(label)
    return unfolded_x, unfolded_y, unfolded_label


def plx_generic_plot(x_var: str, y_var: str, label_var: str, plotting_function: Callable, colors=None, **kwargs):
    # styler_args = inspect.getfullargspec(style_function)
    # style_args = ','.join(
    #     [f"{arg}={'""' if default == '' else default}" for arg, default in zip(styler_args.args, styler_args.defaults)])
    # style_args = style_args.replace(" ", "")
    # if len(style_args) > 0: style_args = "," + style_args
    # @with_signature(f"generic_plot({x_var}, {y_var}, {label_var}{style_args}, colors=None)")

    @plottify()
    @with_signature(f"generic_plot(fig, ax, {x_var}, {y_var}, {label_var}, colors)")
    def generic_plot(fig, ax, **vars2plot):
        # with contextmanager(style_function)(**filter_for_func(style_function, vars2plot)) as (fig, ax):
        x, y, label = unfold(vars2plot[x_var], vars2plot[y_var], vars2plot[label_var])
        data = pd.DataFrame.from_dict({x_var: x, y_var: y, label_var: label})
        plotting_function(data, ax=ax, x=x_var, y=y_var, hue=label_var, palette=vars2plot["colors"])
        ax.set(xlabel=x_var, ylabel=y_var)

    return generic_plot(colors=colors, **kwargs)


plx_lineplot = plx_partial(plx_generic_plot, plotting_function=sns.lineplot)
