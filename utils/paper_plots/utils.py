import matplotlib as mpl
import matplotlib.pyplot as plt


def setup_plotting():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 11
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.2

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["pdf.use14corefonts"] = False


_COLORS = {
    "pg": ("#8A8A8A", "#333333"),  # light gray / dark edge — baseline
    "bespoke": ("#2E86AB", "#1a4d6f"),  # blue — bespoke hints
    "true": ("#4CAF50", "#2e7d32"),  # green — true / optimal hints
}
