
import matplotlib.pyplot as plt
import numpy as np

def scatter_hist(x, y, xlabel, ylabel, title, file_name):
    # remove ax, ax_histx, ax_histy from arguments
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, marker = 'o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    # now determine nice limits by hand:
    binwidth = 1
    xymin = min(np.min(x), np.min(y))
    xymax = max(np.max(x), np.max(y))
    # ymin = min(np.min(y))
    # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    # lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(xymin, xymax + binwidth, binwidth)
    # bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    fig.suptitle(title)
    plt.savefig(file_name, dpi=300)