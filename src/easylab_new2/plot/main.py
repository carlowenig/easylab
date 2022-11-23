def setup_plotting():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "font.family": "serif",
            "font.serif": "cm",
            "mathtext.fontset": "cm",
            "text.usetex": True,
            "axes.grid": True,
            "axes.formatter.use_locale": True,
            "grid.linewidth": 0.2,
            "grid.alpha": 0.5,
            "lines.linewidth": 1,
            "lines.dashed_pattern": (6, 4),
            "text.latex.preamble": (
                "\\usepackage{amsmath} \n"
                "\\usepackage{amssymb} \n"
                "\\usepackage{bm} \n"
                "\\usepackage{siunitx} \n"
                "\\usepackage{physics} \n"
                "\\usepackage{braket} \n"
                "\\usepackage{mathtools} \n"
                "\\usepackage{cancel} \n"
                "\\usepackage{mhchem} \n"
            ),
        }
    )


setup_plotting()
