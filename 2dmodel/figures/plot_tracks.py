
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tracpy
import tracpy.plotting
import netCDF4 as netCDF
import numpy as np
import cmocean.cm as cmo


# drifter tracks
d = netCDF.Dataset('../tracks/tracks.nc')
lonp = d['lonp'][:]; latp = d['latp'][:]

# frehd model output
dm = netCDF.Dataset('../model.nc')
x_psi = dm['x_psi'][:]; y_psi = dm['y_psi'][:]
x_u = dm['x_u'][:]; y_u = dm['y_u'][:]
x_v = dm['x_v'][:]; y_v = dm['y_v'][:]

which = 'vort'  # vort or speed

for i in np.arange(0, dm['u'].shape[0], 20):

    fig = plt.figure(figsize=(14.185,6.1))
    fig.subplots_adjust(right=1.0, top=0.97, left=0.05)
    ax = fig.add_subplot(111)
    u = dm['u'][i].squeeze(); v = dm['v'][i].squeeze()
    if which == 'vort':
        # calculate vorticity while moving onto psi grid from u/v grids
        var = (v[:,:-1] - v[:,1:])/(x_v[:,:-1] - x_v[:,1:]) - (u[:-1,:] - u[1:,:])/(y_u[:-1,:] - y_u[1:,:])
        cmap = cmo.curl
        vmax = 0.7; vmin = -vmax
        label = 'Vertical vorticity [s$^{-1}$]'
        c = '0.2' #'cornflowerblue'
        ac = '0.5'
        walpha = 0.6
        # ec = 'darkorange'
    # need next step for both speed calc and quiver arrows
    u = tracpy.op.resize(u, 0); v = tracpy.op.resize(v, 1)
    if which == 'speed':
        var = np.sqrt(u**2 + v**2)
        cmap = cmo.speed
        vmax = 0.3; vmin = 0
        label = 'Speed [m/s]'
        c = 'purple'  # marker color
        # ec = 'purple'  # edge color
        ac = '0.2'
        walpha = 0.4

    mappable = ax.pcolormesh(x_psi, y_psi, var, cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(mappable, pad=0.02)
    cb.set_label(label)

    ax.quiver(x_psi[::4,::4], y_psi[::4,::4], u[::4,::4], v[::4,::4], scale_units='xy', scale=0.3, alpha=walpha, color=ac)
    ax.axis('tight')
    ax.set_xlabel('along-channel [m]')
    ax.set_ylabel('across-channel [m]')

    # add drifters
    ax.plot(lonp[:,i].T, latp[:,i].T, '.', markersize=4, color=c, alpha=0.7)

    # plot time in tidal cycles
    ax2 = fig.add_axes([0.67, 0.125, 0.15, 0.15])
    ax2.plot(dm['ocean_time'][:], dm['u'][:,0,42,115], 'k')
    ax2.plot(dm['ocean_time'][i], dm['u'][i,0,42,115], 'ro')
    ax2.set_frame_on(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(-0.06, 0.47, 'u', transform=ax2.transAxes)

    fig.savefig(which + '/' + str(i).zfill(4) + '.png', bbox_inches='tight')
    plt.close(fig)
