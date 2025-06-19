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
    filename = path2plot if "." in path2plot.split("/")[-1] else path2plot + "." + format
    yield filename
    plt.savefig(filename, dpi=dpi)
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
                with savefigure(path2figure) as new_filename:
                    with contextmanager(style_function)(**args4style) as (fig, ax):
                        ax.set_title("\n".join([f"{k}: {v}" for k, v in plot_by_vars.items()]))
                        plot_func(fig, ax, **kwargs, **results4plot)
                    paths.append(new_filename)
                if verbose: print("Figure saved in:", new_filename)
            return paths

        return wrapper

    return decorator


# ================================================ #
#                   Premade plots                  #
# ================================================ #
def unfold(x_var, y_var, *args):
    unfolded_vars = [[], []] + [[] for _ in range(len(args))]
    for line in zip(*((x_var, y_var,) + args)):
        assert False, "Problem not solved."
        # TODO: problem here
        are_iterables = list(map(lambda d: isinstance(d, Iterable) and not isinstance(d, str), line[:3]))
        if all(are_iterables):
            for i, e in enumerate(line): unfolded_vars[i].append(e)
        else:
            x, y = line[:2]
            other_vars = [[] for _ in range(len(args))]
            if not all(are_iterables):
                assert are_iterables[0] == are_iterables[1] == True, "x and y should be both iterables"
                assert len(x) == len(y), "x and y should have the same length"
                for i, v in enumerate(line[2:]): other_vars[i] = [v] * len(x)
            assert len(set(map(len, (x, y) + tuple(
                other_vars)))) == 1, "x, y and the rest of the variables must have the same length"
            for i, e in enumerate((x, y) + tuple(other_vars)): unfolded_vars[i] += list(e)

    return unfolded_vars


style_properties = ["colors", "sizes", "style", "markers"]


def plx_generic_plot(x_var: str, y_var: str, label_var: str, plotting_function: Callable, reduce: Callable = None,
                     **kwargs):
    @plottify()
    @with_signature(
        f"generic_plot(fig, ax, {x_var}, {y_var}, {label_var}, {', '.join([f'{pn}=None' for pn in style_properties])})")
    def generic_plot(fig, ax, **vars2plot):

        # Transform style variables from dict to list
        for property_name in set(style_properties).intersection(vars2plot):
            property = vars2plot[property_name]
            if property is None:
                vars2plot.pop(property_name)
            else:
                if isinstance(property, dict):
                    if len(set(vars2plot[label_var]).difference(property.keys())) == 0:
                        vars2plot[property_name] = [vars2plot[property_name][k] for k in vars2plot[label_var]]
                    else:
                        raise AssertionError(
                            f"{property_name} should have keys {set(vars2plot[label_var])} but has only {property.keys()}")

        # Prepare data for plot
        vars2plot = filter_dict(vars2plot, keys=[x_var, y_var, label_var] + style_properties)
        data_values = unfold(*vars2plot.values())
        data = pd.DataFrame(dict(zip(vars2plot, data_values)))
        # x, y, label = unfold(vars2plot[x_var], vars2plot[y_var], vars2plot[label_var])
        # data = pd.DataFrame.from_dict({x_var: x, y_var: y, label_var: label})

        # # Add style variables
        # for property_name in style_properties:
        #     property = vars2plot[property_name]
        #     if property is not None:
        #         if isinstance(property, list):
        #             if len(property) == len(label):
        #                 data[property_name] = vars2plot[property_name]
        #             else:
        #                 raise ValueError(
        #                     f"property {property_name} has {len(property)} elements and should have {len(label)} elements.")
        #         elif isinstance(property, dict):
        #             if len(set(vars2plot[label_var]).difference(property.keys())) == 0:
        #                 data[property_name] = [vars2plot[property_name][k] for k in data[label_var]]
        #             else:
        #                 raise AssertionError(
        #                     f"{property_name} should have keys {set(vars2plot[label_var])} but has only {property.keys()}")

        # Apply reduce over y_var variable
        if reduce is not None:
            data = data.groupby(data.columns[~data.columns.isin([y_var])].to_list()).apply(reduce)
            data.name = y_var
            data = data.reset_index()

        # TODO: dashes etc not working as expected.
        style_properties_dict = data.rename(columns={"colors": "palette"}).loc[:,
                                data.columns.isin(style_properties)].to_dict(orient='list')
        plotting_function(data=data, ax=ax, x=x_var, y=y_var, hue=label_var, **style_properties_dict)
        ax.set(xlabel=x_var, ylabel=y_var)

    return generic_plot(**kwargs)


plx_lineplot = plx_partial(plx_generic_plot, plotting_function=sns.lineplot)
