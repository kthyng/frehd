'''
read in mat files, initialized drifter sim, and run
'''

# up to 7.1
from scipy.io import loadmat
# up to 7.3
import h5py
import xarray as xr
import tracpy
import octant
import os
from datetime import datetime, timedelta
import numpy as np
from tracpy.tracpy_class import Tracpy
import netCDF4 as netCDF


startdate = datetime(0001, 1, 1, 0, 0, 0, 0)
dt = 0.05  # seconds, time step
time_units = 'seconds since 0001-01-01  00:00:00'
# dz = 0.005  # meters, layer thickness
# s_rho = np.array([ 0.0025,  0.0075,  0.0125,  0.0175,  0.0225,  0.125 ])

def setup():
    '''Read in .mat files. Convert to something I can use.

    apparently the files were made in Matlab version up to 7.3

    From Katie:

    It is a 3 cm water depth case with a 3rd order upwind scheme and calibrated
    bottom friction and turbulence variables. The simulation matches the PIV
    data well in terms of vortex diameter, circulation, and location. I included
    all of the 3D model outputs including: x, y, and z velocities, x, y, and z
    viscosities, and water depth. The time step for this data is 0.05 seconds
    and it was run for a total of 4401 steps. There is an individual variable
    for each time step in this data. The grid cells at the bottom of the tank
    are represented by xVelocity(:,:,1) and the grid cells at the surface are
    represented by either xVelocity(:,:,6) or xVelocity(:,:,7) depending on the
    water depth at that grid cell. The grid is 85x232x8. The horizontal grid
    size is (5.5/85)x(15/232) m and the vertical grid thickness is 0.005 m.

    OLD:
    I have attached the y-velocity and x-velocity structures for the FREHD case with the best results. This case is a model run with a 6.5x6.5 cm grid size, theta predictor-corrector scheme, non-hydrostatic terms turned off, sidewall and bottom friction turned on, and a one equation turbulence model applied. It was run for an idealized inlet with a 5 cm water depth. The entire domain covers 5.5x15 m. The y-direction is horizontal and is in the direction of flow and x-direction is vertical. I also attached a schematic of the tank for the laboratory setup and model setup in case this is unclear. The model was run with a 0.01 s time step but I only output every 10th step. It was run for a total of 5 tidal cycles with a 50 s tidal period for each tidal cycle. I also attached structures for the average velocity through the inlet and the model time associated. I included the calculated vorticity as well for reference.

    The following is from the PIV simulations but might still be true:
    The total time for the simulations is 220 seconds for a total of 4 tidal cycles. Any analysis I do, I ignore the first tidal cycle since the behavior was proven to be different than the other 3 so drifters need to be run for a minimum of 2 tidal cycles or about 110 seconds.
    '''

    # create time array: time step is 0.05 seconds, 4401 steps
    t = [startdate + timedelta(seconds=dt)*i for i in range(4401)]
    t = netCDF.date2num(t, time_units)

    # grid
    g = loadmat('model/out_0_3D.mat')
    # swapping katie's x and y now so rest should be normal
    x_rho = g['yy'][:,:,0]
    y_rho = g['xx'][:,:,0]

    # zz = g['zz'][0,0,:]  # ??? doesn't add up to 3 cm for one thing...
    # s_w = np.linspace(-1, 0, zz.size+1)
    # s_rho = tracpy.op.resize(s_w, 0)
    # Cs_w = np.linspace(-1, 0, s_rho.size+1)

    # array sizes
    lk = 1  # s_rho.shape[0]  # vertical
    ly = x_rho.shape[0]  # across-channel
    lx = x_rho.shape[1]  # along-channel

    v_rho = np.zeros((len(t), lk, ly, lx))
    u_rho = np.zeros((len(t), lk, ly, lx))
    zeta = np.zeros((len(t), ly, lx))  # free surface

    # loop through time and read in output
    for i, tstep in enumerate(t):

        if i == 0:
            continue

        # across-channel velocity: vertical x across x along (after axes are swapped)
        v_rho[i] = h5py.File('model/out_' + str(i) + '_xVelocity_3D.mat')['xVelocity'][-1, :].swapaxes(0, 1)

        # along-channel velocity: vertical x across x along (after axes are swapped)
        u_rho[i] = h5py.File('model/out_' + str(i) + '_yVelocity_3D.mat')['yVelocity'][-1, :].swapaxes(0, 1)

        # free surface
        zeta[i] = h5py.File('model/out_' + str(i) + '_surfaceZ_3D.mat')['surfaceZ'][:].T

    # create other grid variables
    angle = np.zeros(x_rho.shape)
    dx = np.diff(x_rho)[0][0]; dy = np.diff(y_rho)[0][0]
    pm = (1./dx)*np.ones(x_rho.shape); pn = (1./dy)*np.ones(y_rho.shape)
    x_vert, y_vert = octant.grid.rho_to_vert(x_rho, y_rho, pm, pn, angle)
    grid = octant.grid.CGrid(x_vert, y_vert)

    # interpolate to u/v grids for velocity
    u = tracpy.op.resize(u_rho, 3)
    v = tracpy.op.resize(v_rho, 2)

    # skip ahead by indices input in diff to align with drifter simulation
    # MAYBE NOT THIS PART?
    # u = u[diff:]; v = v[diff:]; t = t[0,diff:]; zeta = zeta[diff:]
    # import pdb; pdb.set_trace()
    # set up to write to netcdf file
    ds = xr.Dataset({'u': (['ocean_time', 's_rho', 'eta_u', 'xi_u'], u),
                     'v': (['ocean_time', 's_rho',  'eta_v', 'xi_v'], v),
                     'zeta': (['ocean_time', 'eta_rho', 'xi_rho'], zeta),
                     'mask': (['eta_rho', 'xi_rho'], grid.mask),
                     'h': (['eta_rho', 'xi_rho'], 0.03*np.ones(grid.x_rho.shape)),
                     'hc': 0,
                     'theta_s': 0.0001,
                     'theta_b': 0},
                    coords = {'ocean_time': t,
                              's_rho': ('s_rho', [-0.5]),
                              's_w': ('s_w', [-1, 0]),
                              'Cs_w': ('Cs_w', [-1, 0]),
                              'y_vert': (['eta_vert', 'xi_vert'], grid.y_vert),
                              'x_vert': (['eta_vert', 'xi_vert'], grid.x_vert),
                              'y_psi': (['eta_psi', 'xi_psi'], grid.y_psi),
                              'x_psi': (['eta_psi', 'xi_psi'], grid.x_psi),
                              'y_rho': (['eta_rho', 'xi_rho'], grid.y_rho),
                              'x_rho': (['eta_rho', 'xi_rho'], grid.x_rho),
                              'y_u': (['eta_u', 'xi_u'], grid.y_u),
                              'x_u': (['eta_u', 'xi_u'], grid.x_u),
                              'y_v': (['eta_v', 'xi_v'], grid.y_v),
                              'x_v': (['eta_v', 'xi_v'], grid.x_v)})
    ds['ocean_time'].attrs['units'] = time_units

    ds.to_netcdf('model/model.nc', format='NETCDF4')


def init(name):
    '''Initialize tracpy simulation'''

    time_units = 'seconds since 0001-01-01  00:00:00'

    nsteps = 25

    # Number of steps to divide model output for outputting drifter location
    N = 1

    # Number of days
    # 165 seconds max or less as simulations step forward in time
    # name accounts for start time stepping forward in time each simulation
    if len(name) == 19:
        startdate = datetime.strptime(name, '%Y-%m-%dT%H:%M:%S')
    elif len(name) > 19:
        startdate = datetime.strptime(name, '%Y-%m-%dT%H:%M:%S.%f')
    starttime = netCDF.date2num(startdate, time_units)
    ndays = (220. - starttime - dt)/86400

    # This is a forward-moving simulation
    ff = 1

    # Time between outputs
    tseas = dt  # time between output in seconds
    ah = 0.
    av = 0.  # m^2/s

    # surface drifters
    z0 = 's'
    zpar = 0  # 1 layer

    # for 3d flag, do3d=0 makes the run 2d and do3d=1 makes the run 3d
    do3d = 0
    doturb = 0

    # for periodic boundary conditions in the x direction
    doperiodic = 0

    # Flag for streamlines. All the extra steps right after this are for streamlines.
    dostream = 0
    # import pdb; pdb.set_trace()

    proj = tracpy.tools.make_proj(setup='galveston')
    loc = 'model/model.nc'
    grid = tracpy.inout.readgrid(loc, proj, usespherical=False)

    # Start uniform array of drifters across domain using x,y coords
    # z0temp = tracpy.op.resize(np.linspace(-1,0,7), 0)  # s_rho
    # z0 = np.concatenate((z0temp, z0temp))
    # x0temp = [7.435, 7.435]
    # x0 = np.concatenate((x0temp[0]*np.ones(z0temp.size), x0temp[1]*np.ones(z0temp.size)))
    # y0temp = [2.3945, 2.968]
    # y0 = np.concatenate((y0temp[0]*np.ones(z0temp.size), y0temp[1]*np.ones(z0temp.size)))
    x0 = [7.435, 7.435]
    y0 = [2.3945, 2.968]

    # Initialize Tracpy class
    tp = Tracpy(loc, grid, name='tracks/tracks_' + name, tseas=tseas,
                ndays=ndays, nsteps=nsteps, N=N, ff=ff, ah=ah, av=av,
                doturb=doturb, do3d=do3d, z0=z0, zpar=zpar,
                time_units=time_units, usespherical=False, savell=True,
                doperiodic=doperiodic)

    # force grid reading
    # tp._readgrid()

    return tp, x0, y0


def run():

    # Make sure necessary directories exist
    if not os.path.exists('tracks'):
        os.makedirs('tracks')
    if not os.path.exists('figures'):
        os.makedirs('figures')

    thisstartdate = datetime(0001, 1, 1, 0, 0, 55, 0)  # start for this simulation
    timeleft = 165  # seconds
    while thisstartdate < thisstartdate + timedelta(seconds=timeleft - 1):

        # initialdate = datetime(0001, 1, 1, 0, 0, 0, 10000)  # earliest model time
        # diff = (thisstartdate - startdate).microseconds
        # name = str(diff)
        name = thisstartdate.isoformat()  # e.g. '0001-01-01T00:00:55'

        fmodel = 'model/model.nc'
        if not os.path.exists(fmodel):
            setup()#diff/dt)  # send in number of indices to skip ahead

        # If the particle trajectories have not been run, run them
        if not os.path.exists('tracks/tracks_' + name + '.nc') and \
           not os.path.exists('tracks/tracks_' + name + 'gc.nc'):

            # Read in simulation initialization
            tp, x0, y0 = init(name)

            xp, yp, zp, t, T0, U, V = tracpy.run.run(tp, thisstartdate, x0, y0)

        # Increment by 24 hours for next loop, to move through more quickly
        thisstartdate += timedelta(seconds=0.05)



if __name__ == "__main__":
    run()
