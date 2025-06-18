import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Tuple, Dict, Any, List, Union

import pandas as pd
import seaborn as sns
from makefun import with_signature
from matplotlib import pyplot as plt

from perplexitylab.experiment_tools import ExperimentManager
from perplexitylab.miscellaneous import filter_dict, filter_for_func, plx_partial, group


# ================================================ #
#                    Plot tools                    #
# ================================================ #
@contextmanager
def savefigure(path2plot, format="png", dpi=None):
    Path(path2plot).parent.mkdir(parents=True, exist_ok=True)
    yield
    plt.savefig(path2plot if "." in path2plot else path2plot + "." + format, dpi=dpi)
    plt.close()


# ================================================ #
#          Connect plot with experiment            #
# ================================================ #
def plx_generic_plot_styler(figsize=(8, 6), log="",
                            xlabel=None, xlabel_fontsize=None, xticks=None, xticks_labels=None, xlim=None,
                            ylabel=None, ylabel_fontsize=None, yticks=None, yticks_labels=None, ylim=None,
                            legend_fontsize=None, legend_loc=None, bbox_to_anchor=None):
    def styler_function():
        fig, ax = plt.subplots(figsize=figsize)

        yield fig, ax

        if xlabel is not None: ax.set_xlabel(xlabel)
        if xlabel_fontsize is not None: ax.set_xlabel(ax.get_xlabel(), fontsize=xlabel_fontsize)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if ylabel_fontsize is not None: ax.set_ylabel(ax.get_ylabel(), fontsize=ylabel_fontsize)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if "x" in log: ax.set_xscale("log")
        if "y" in log: ax.set_yscale("log")
        if xticks is not None: ax.set_xticks(xticks, xticks)
        if yticks is not None: ax.set_yticks(yticks, yticks)
        if xticks_labels is not None: ax.set_xticklabels(xticks_labels)
        if yticks_labels is not None: ax.set_xticklabels(yticks_labels)
        if legend_fontsize is not None or legend_loc is not None or bbox_to_anchor is not None:
            ax.legend(fontsize=legend_fontsize,
                      bbox_to_anchor=bbox_to_anchor,
                      loc=legend_loc)
        fig.tight_layout()

    return styler_function


def plottify(variables_assumed_unique=()):
    def decorator(plot_func):
        def wrapper(
                em: ExperimentManager, filename: str,
                experiment_setup: Union[
                    Tuple[Dict[str, Any], Dict[str, Any]], List[Tuple[Dict[str, Any], Dict[str, Any]]]],
                common_experiment_setup: Tuple[Dict[str, Any], Dict[str, Any]] = ({}, {}),
                folder="", path=None, verbose=True, plot_by=(),
                style_function=plx_generic_plot_styler(), **kwargs):
            args4style = filter_for_func(style_function, kwargs)
            plot_func_variables = inspect.getfullargspec(plot_func).args
            assert "fig" in plot_func_variables, f"the input variable 'fig' should be in {plot_func.__name__} even if is unused."
            assert "ax" in plot_func_variables, f"the input variable 'ax' should be in {plot_func.__name__} even if is unused."
            # required_variables = set(plot_func_variables).intersection(variables)
            results = em.run_experiments(experiment_setup=experiment_setup,
                                         required_variables=plot_func_variables + list(plot_by),
                                         common_experiment_setup=common_experiment_setup)

            paths = []
            for i, (results4plot, plot_by_vars) in enumerate(group(results, *plot_by)):
                results4plot = filter_for_func(plot_func, results4plot)  # get only variables relevant for plot
                kwargs = filter_for_func(plot_func, kwargs)  # get only params relevant for plot
                kwargs = filter_dict(kwargs,
                                     keys_not=list(results4plot.keys()))  # get params for plot (not already in results)
                for k in variables_assumed_unique:
                    results4plot[k] = em.constants[k] if k in em.constants else results4plot[k].pop()
                path2figure = f"{path if path is not None else em.path}/{folder}/{filename}_{i}"
                with savefigure(path2figure):
                    with contextmanager(style_function)(**args4style) as (fig, ax):
                        ax.set_title("\n".join([f"{k}: {v}" for k, v in plot_by_vars.items()]))
                        plot_func(fig, ax, **kwargs, **results4plot)
                if verbose: print("Figure saved in:", path2figure)
                paths.append(path2figure)
            return paths

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


def plx_generic_plot(x_var: str, y_var: str, label_var: str, plotting_function: Callable, reduce: Callable = None,
                     **kwargs):
    @plottify()
    @with_signature(
        f"generic_plot(fig, ax, {x_var}, {y_var}, {label_var}, colors=None, sizes=None, dashes=None, markers=None)")
    def generic_plot(fig, ax, **vars2plot):
        x, y, label = unfold(vars2plot[x_var], vars2plot[y_var], vars2plot[label_var])
        data = pd.DataFrame.from_dict({x_var: x, y_var: y, label_var: label})
        if reduce is not None:
            data = data.groupby([x_var, label_var]).apply(reduce)
            data.name = y_var
            data = data.reset_index()
        #TODO: dashes etc not working as expected.
        plotting_function(data=data, ax=ax, x=x_var, y=y_var, hue=label_var, palette=vars2plot["colors"],
                          sizes=vars2plot["sizes"], dashes=vars2plot["dashes"], markers=vars2plot["markers"])
        ax.set(xlabel=x_var, ylabel=y_var)

    return generic_plot(**kwargs)


plx_lineplot = plx_partial(plx_generic_plot, plotting_function=sns.lineplot)
