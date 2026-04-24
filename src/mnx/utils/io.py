import numpy as np
import matplotlib.pyplot as plt

def plot_bands(ax, qpath, bands, xticks, xlabels, color = "tab:blue", lw = 1.5, alpha=0.8, label=None):
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

def plot_bands_segment(qpath, bands, data, xticks, xlabels):
    
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm, ListedColormap
    from scipy.interpolate import interp1d
    
    f = plt.figure()
    ax = f.add_subplot()

    n_bins = 256
    base_cmap = plt.get_cmap("plasma")
    colors_with_alpha = base_cmap(np.linspace(0, 1, n_bins))

    # Create an alpha ramp (0 to 1) and square it to match your "alphas**2" logic
    alpha_ramp = np.linspace(0, 1, n_bins)**2
    colors_with_alpha[:, -1] = alpha_ramp  # Set the 4th column (Alpha)
    alpha_plasma = ListedColormap(colors_with_alpha)


    num_interp = 4000 
    x_interp = np.linspace(qpath[0], qpath[-1], num_interp)
    norm = LogNorm(1e-2, 1)

    # Background faint lines
    for mode in range(12):
        ax.plot(qpath, np.real(bands[:, mode]), color="#0d0887", alpha=0.1)

    # Main Line Collections
    for mode in range(12):
        # Get and interpolate data
        y_raw = np.real(np.array(bands[:, mode]))
        z_raw = np.abs(data[:, mode])
         
        f_y = interp1d(qpath, y_raw, kind='linear')
        f_z = interp1d(qpath, z_raw, kind='linear')
        
        y = f_y(x_interp)
        z = f_z(x_interp)
        
        # Create segments
        points = np.array([x_interp, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        z_segments = z[:-1]
        
        # Use the CUSTOM alpha_plasma cmap here
        lc = LineCollection(segments, cmap=alpha_plasma, norm=norm)
        
        # We no longer strictly need lc.set_alpha() because the cmap handles it,
        # but setting the array triggers the color/alpha mapping
        lc.set_array(z_segments)
        lc.set_linewidth(1)
        lc.set_capstyle('round')
        
        line = ax.add_collection(lc)

    # --- 3. Styling & Colorbar ---
    ax.set_xlim(qpath[0], qpath[-1])
    ax.set_ylim(bands.min()-10, bands.max()+10), ax.set_xlim(0, qpath[-1])
    ax.hlines(y=0, xmin=0, xmax=qpath[-1], color="grey", linestyle="dashed", lw=0.5)
    ax.vlines(x=xticks, ymin=-100000, ymax=100000, color="grey", linestyle="dashed", lw=0.5)
    ax.set_xticks(xticks, labels=xlabels)

    # The colorbar now inherits the transparency from alpha_plasma
    cbar = f.colorbar(line, ax=ax)
    cbar.set_label('Dynamical structure-factor (a.u.)', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    # Ensure the colorbar looks smooth
    cbar.solids.set(alpha=None) 

    ax.tick_params(labelsize=12)


def map_from_bands(ax, qpath, bands, xticks, xlabels, data, Ny=1000, sigma=10, vmin=0, vmax=None):
    if vmax == None:
        vmax = np.real(data.max())
    grid_x = qpath
    grid_y = np.arange(0, np.real(bands.max())+20, (np.real(bands.max())+20)/Ny)
    mmap = np.zeros([len(grid_x), len(grid_y)], dtype=float)
    for xi, x in enumerate(grid_x):
        for yi, y in enumerate(grid_y):
            for mode in range(12):
                mmap[xi,yi] += gaussian(np.real(bands[xi, mode])-y, sigma=sigma)*data[xi,mode]
    #ax = plot_bands(ax, qpath, bands, xticks, xlabels)
    ax.imshow(mmap.T, interpolation="gaussian", origin="lower", aspect="auto",  extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], vmin=0, vmax=vmax, cmap="plasma")
    plot_bands(ax,qpath,bands,xticks,xlabels,color="white",alpha=0.2)
    ax.set_ylim(0,(np.real(bands.max())+20))
    # ax.set_xticks(xticks,xlabels)
    # ax.set_ylabel("$\omega$ (cm$^{-1}$)", fontsize=12)
    # ax.tick_params(labelsize=12)
    return(ax, mmap.T, grid_x, grid_y)

def gaussian(freq, sigma):
    return 1/(sigma*np.sqrt(np.pi*2))*np.exp(-0.5*(np.conj(freq)*freq)/(sigma)**2)