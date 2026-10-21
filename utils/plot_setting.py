"""Functions that set up the plotting style."""

import matplotlib.pyplot as plt

def set_default_plots(figsize=(6.06, 8), dpi: int = 300):
    """
    Set the parameters for the visualization of plots to predetermined size, style and resolution.

    Args:
        figsize (tuple(int,int)): The size of the plots in Inches. Defaults to (10,10).

        dpi (int): The resolution of the plots in dots per square inch. Defaults to 100.
    """

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["rm-lmr10"],
            "font.size": 7,
            "figure.facecolor": "white",
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.5,
            "lines.markersize": 2.5,
            "lines.linewidth": 0.7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            # "xtick.labelsize": 8,
            "xtick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            # "ytick.labelsize": 8,
            "legend.fontsize": 6,
            "legend.fancybox": False,
            "savefig.format": "pdf",
            "axes.grid": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # plt.rc("figure", figsize=golden_ratio(), dpi=dpi)
    plt.rc("figure", figsize=figsize, dpi=dpi)
