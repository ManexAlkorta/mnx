import numpy as np

def plot_bands(ax, qpath, bands, xticks, xlabels, color = "tab:blue", lw = 1.5, alpha=0.8, label=None):
    for mode in range(len(bands[0,:])):
        if np.square(bands[:,mode]).min()<0:
            for qi in range(len(qpath)):
                if np.square(bands[qi,mode])<0:
                    bands[qi,mode] = (-1)*np.sqrt(np.abs(np.square(bands[qi,mode])))
    bands = np.real(bands)
    for mode in range(len(bands[0,:])):
        if label != None:
            if mode == 0:
                ax.plot(qpath, np.real(bands[:, mode]), color=color, alpha=alpha, lw = lw, label=label)
            else:
                ax.plot(qpath, np.real(bands[:, mode]), color=color, alpha=alpha, lw = lw)
        else:
            ax.plot(qpath, np.real(bands[:, mode]), color=color, alpha=alpha, lw = lw)
    ax.set_ylabel(r"$\omega$ (cm$^{-1}$)", fontsize=12)
    ax.set_ylim(bands.min()-10, bands.max()+10), ax.set_xlim(0, qpath[-1])
    ax.tick_params(labelsize=12)
    ax.hlines(y=0, xmin=0, xmax=qpath[-1], color="grey", linestyle="dashed", lw=0.5)
    ax.vlines(x=xticks, ymin=-100000, ymax=100000, color="grey", linestyle="dashed", lw=0.5)
    ax.set_xticks(xticks, labels=xlabels)

