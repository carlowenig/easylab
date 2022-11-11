from typing import Iterable, Optional, TypeVar, cast
from matplotlib import pyplot as plt, ticker, axes
import locale
from ..data import data, value
from ..lang import lang


_X = TypeVar("_X")
_Y = TypeVar("_Y")


def init_matplotlib():
    # Set to German locale to get comma decimal separater
    locale.setlocale(locale.LC_NUMERIC, "de_DE")

    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "figure.figsize": (7, 4),
            "font.family": "serif",
            "font.serif": "cm",
            "mathtext.fontset": "cm",
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{icomma}",
            "axes.grid": True,
            "axes.labelsize": 12,
            "axes.formatter.use_locale": True,
            "grid.linewidth": 0.2,
            "grid.alpha": 0.5,
            "lines.linewidth": 1,
            "lines.dashed_pattern": (6, 4),
        }
    )


def plot(
    x_var: "data.Var[_X]",
    y_var: "data.Var[_Y]",
    value_pairs: Iterable[tuple[_X, _Y]],
    *args,
    ax: Optional[axes.Axes] = None,
    method: Optional[str] = None,
    **kwargs
):
    if ax is None:
        ax = cast(axes.Axes, plt.gca())

    x_vals = []
    y_vals = []
    x_errs = []
    y_errs = []

    for x_val, y_val in value_pairs:
        # if isinstance(x_val, value.Value):
        #     x_val = x_val.ununit()

        # if isinstance(y_val, value.Value):
        #     y_val = y_val.ununit()

        x_vals.append(x_var.output(x_val, "plot", parse=False))
        y_vals.append(y_var.output(y_val, "plot", parse=False))

        x_errs.append(x_var.output(x_val, "plot_err", parse=False))
        y_errs.append(y_var.output(y_val, "plot_err", parse=False))

    if any(err > 1e-10 for err in x_errs + y_errs) and method is None:
        method = "errorbar"
        kwargs |= {"xerr": x_errs, "yerr": y_errs}
    else:
        method = method or "plot"

    result = getattr(ax, method)(x_vals, y_vals, *args, **kwargs)

    # display_x_var = (
    #     x_var.remove_unit().remove_err() if isinstance(x_var, value.ValueVar) else x_var
    # )
    # display_y_var = (
    #     y_var.remove_unit().remove_err() if isinstance(y_var, value.ValueVar) else y_var
    # )

    x_label = x_var.text
    if isinstance(x_var, value.ValueVar):
        x_label += lang.space + "/" + lang.space + x_var.unit.text

    y_label = y_var.text
    if isinstance(y_var, value.ValueVar):
        y_label += lang.space + "/" + lang.space + y_var.unit.text

    # Set axis labels
    ax.set_xlabel(lang.math(x_label).latex)
    ax.set_ylabel(lang.math(y_label).latex)

    def remove_unit_and_err(val):
        if isinstance(val, value.Value):
            return val.remove_unit().remove_err()
        else:
            return val

    # Set tick formatters
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: lang.math(
                x_var.format(
                    remove_unit_and_err(x_var.parse(x, check=False)),
                    parse=False,
                    check=False,
                )
            ).latex
        )
    )
    ax.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(
            lambda y, _: lang.math(
                y_var.format(
                    remove_unit_and_err(y_var.parse(y, check=False)),
                    parse=False,
                    check=False,
                )
            ).latex
        )
    )

    if kwargs.get("label") not in [None, ""]:
        ax.legend()

    plt.tight_layout()

    return result
