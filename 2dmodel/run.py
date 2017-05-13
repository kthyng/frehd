'''
read in mat files, initialized drifter sim, and run
'''

from scipy.io import loadmat
import xarray as xr
import tracpy
import octant
import os
from datetime import datetime
import numpy as np
from tracpy.tracpy_class import Tracpy


def setup(diff):
    '''Read in .mat files.

    apparently the files were made in Matlab version up to 7.1

    From Katie:
    I have attached the y-velocity and x-velocity structures for the FREHD case with the best results. This case is a model run with a 6.5x6.5 cm grid size, theta predictor-corrector scheme, non-hydrostatic terms turned off, sidewall and bottom friction turned on, and a one equation turbulence model applied. It was run for an idealized inlet with a 5 cm water depth. The entire domain covers 5.5x15 m. The y-direction is horizontal and is in the direction of flow and x-direction is vertical. I also attached a schematic of the tank for the laboratory setup and model setup in case this is unclear. The model was run with a 0.01 s time step but I only output every 10th step. It was run for a total of 5 tidal cycles with a 50 s tidal period for each tidal cycle. I also attached structures for the average velocity through the inlet and the model time associated. I included the calculated vorticity as well for reference.

    The total time for the simulations is 220 seconds for a total of 4 tidal cycles. Any analysis I do, I ignore the first tidal cycle since the behavior was proven to be different than the other 3 so drifters need to be run for a minimum of 2 tidal cycles or about 110 seconds.
    '''


    t = loadmat('modeltime.mat')['modeltime']

    # u is in the x direction which is across the channel
    # time x x (across) x y (along)
    f = loadmat('xVel.mat')['xVel'][0][0]
    v_rho = np.zeros((len(f), 1, f[0].shape[0], f[0].shape[1]))
    for i in range(len(f)):
        v_rho[i,0] = f[i]

    # v is in the y direction which is along the channel
    f = loadmat('yVel.mat')['yVel'][0][0]
    u_rho = np.zeros((len(f), 1, f[0].shape[0], f[0].shape[1]))
    for i in range(len(f)):
        u_rho[i,0] = f[i]

    # back out x and y rho grid locations
    ly = 5.5  # m, width of channel (Katie's x-direction)
    lx = 15  # m, length of channel (Katie's y-direction)
    y_rho = np.linspace(0, ly, u_rho.shape[2])
    x_rho = np.linspace(0, lx, u_rho.shape[3])
    Y, X = np.meshgrid(y_rho, x_rho)
    Y = Y.T; X = X.T;

    # create other grid variables
    angle = np.zeros(X.shape)
    dx = np.diff(x_rho)[0]; dy = np.diff(y_rho)[0]
    pm = (1./dx)*np.ones(X.shape); pn = (1./dy)*np.ones(Y.shape)
    x_vert, y_vert = octant.grid.rho_to_vert(X, Y, pm, pn, angle)
    grid = octant.grid.CGrid(x_vert, y_vert)


    # interpolate to u/v grids for velocity
    u = tracpy.op.resize(u_rho, 3)
    v = tracpy.op.resize(v_rho, 2)

    # skip ahead by indices input in diff to align with drifter simulation
    u = u[diff:]; v = v[diff:]; t = t[0,diff:]
    # import pdb; pdb.set_trace()
    # set up to write to netcdf file
    ds = xr.Dataset({'u': (['ocean_time', 's_rho', 'eta_u', 'xi_u'], u),
                     'v': (['ocean_time', 's_rho',  'eta_v', 'xi_v'], v),
                     'mask': (['eta_rho', 'xi_rho'], grid.mask),
                     'h': (['eta_rho', 'xi_rho'], 0.05*np.ones(grid.x_rho.shape)),
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
    time_units = 'seconds since 0001-01-01  00:00:00'
    ds['ocean_time'].attrs['units'] = time_units

    ds.to_netcdf('model.nc', format='NETCDF4')


def init(name):
    '''Initialize tracpy simulation'''

    time_units = 'seconds since 0001-01-01  00:00:00'

    nsteps = 25

    # Number of steps to divide model output for outputting drifter location
    N = 1

    # Number of days
    ndays = 150./86400  # 150 sec  0.0025  # 220 seconds

    # This is a forward-moving simulation
    ff = 1

    # Time between outputs
    tseas = 0.1  # time between output in seconds
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
    loc = 'model.nc'
    grid = tracpy.inout.readgrid(loc, proj, usespherical=False)

    # Initialize Tracpy class
    tp = Tracpy(loc, grid, name='tracks', tseas=tseas, ndays=ndays, nsteps=nsteps,
                N=N, ff=ff, ah=ah, av=av, doturb=doturb, do3d=do3d, z0=z0, zpar=zpar, time_units=time_units,
                usespherical=False, savell=True, doperiodic=doperiodic)

    # force grid reading
    # tp._readgrid()

    # Start uniform array of drifters across domain using x,y coords
    x0 = grid.x_rho[1:-1:2,1:-1:2]
    y0 = grid.y_rho[1:-1:2,1:-1:2]

    return tp, x0, y0


def run():

    # Make sure necessary directories exist
    if not os.path.exists('tracks'):
        os.makedirs('tracks')
    if not os.path.exists('figures'):
        os.makedirs('figures')

    initialdate = datetime(0001, 1, 1, 0, 0, 0, 10000)  # earliest model time
    startdate = datetime(0001, 1, 1, 0, 1, 6, 10000)  # start for this simulation
    diff = (startdate-initialdate).seconds
    name = str(diff)

    setup(diff/0.1)  # send in number of indices to skip ahead

    # If the particle trajectories have not been run, run them
    if not os.path.exists('tracks/' + name + '.nc') and \
       not os.path.exists('tracks/' + name + 'gc.nc'):

        # Read in simulation initialization
        tp, x0, y0 = init(name)

        xp, yp, zp, t, T0, U, V = tracpy.run.run(tp, startdate, x0, y0)

    # # Increment by 24 hours for next loop, to move through more quickly
    # date = date + timedelta(hours=24)



if __name__ == "__main__":
    run()
