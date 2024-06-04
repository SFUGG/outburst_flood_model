"""
Solvers for compressible and incompressible outburst flood conduit
models
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate

# PHYSICAL CONSTANTS (can't change these values)
rhoi = 917          # Density of ice (kg.m-3)
rhow = 1000         # Density of water (kg.m-3)
g = 9.81            # Gravitational acceleration (m.s-2)
Lf = 3.34e5         # Latent heat of fusion (J.kg-1)
cw = 4.217e3        # Heat capacity of water (J.kg-1.K-1
ct = 7.5e-8         # Clausius-Clapeyron (pressure-melting) constant (K.Pa-1)
gamma = ct*rhow*cw  # Derived coefficient in energy-balance equation

# PARAMETERS - can change these vaues
default_params = dict(
A = 2.4e-24,        # Flow law coefficient (Pa-3.s-1)
n = 3,              # Flow law exponent
fR = 0.15,          # Darcy-Weisbach friction coefficient
beta = 1e-4,        # Compressibility (Pa-1)
)

# HELPER FUNCTIONS
def wrap(S, pw, h):
    """
    Concatenate S, pw, and h arrays into the state vector
    """
    N = len(S)
    v = np.zeros(2*N+1)
    v[:N] = S
    v[N:2*N] = pw
    v[-1] = h
    return v

def unwrap(v):
    """
    Split state vector into arrays S, pw, and h
    """
    N = int(len(v-1)/2)
    S = v[:N]
    pw = v[N:2*N]
    h = v[-1]
    return S, pw, h

def rhs_compressible_channel(t, v, params):
    """
    Calculate d(pw)/dt and d(S)/dt for nonsteady, compressible conduit
    equations.
    """
    S, pw, h = unwrap(v)

    A = params['A']
    n = params['n']
    fR = params['fR']
    beta = params['beta']

    dx = params['dx']
    N = params['N']

    p_i = params['p_i']

    A_lake = params['A_lake']
    p_lake = rhow*g*h
    # Compute pressure gradient at lake outlet
    pgrad_lake = (pw[0] - p_lake)/dx

    if params['drainage']=='pressure-coupled':
        # Compute Q discharge from lake (avoiding divide by zero error)
        if pgrad_lake!=0:
            Q_lake = -S[0]**(5/4)*(1/rhow/fR)**(1/2)*(1/np.pi)**(1/4)*np.abs(pgrad_lake)**(-0.5)*pgrad_lake
        else:
            Q_lake = 0

        # Do not allow negative discharge (e.g. lake filling from conduit)
        if Q_lake<0:
            Q_lake = 0

    else:
        Q_lake_fun = params['Q_lake']
        Q_lake = Q_lake_fun(t)

    # Stop draining when h is <= 0
    if h<=0:
        Q_lake = 0

    N = params['N']
    # DERIVATIVE OPERATORS
    # Upwind derivatives - this enforces a BC on the left boundary
    D_upwind = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
    D_upwind = D_upwind/dx

    # Downwind derivatives - this enforces a BC on the right boundary
    D_downwind = 1/dx*(-np.diag(np.ones(N)) + np.diag(np.ones(N-1), k=1))

    dpwds = np.matmul(D_downwind, np.vstack(pw)).flatten()
    Q = -S**(5/4)*(1/rhow/fR)**(1/2)*(1/np.pi)**(1/4)*np.abs(dpwds)**(-0.5)*dpwds
    dQds = np.matmul(D_upwind, np.vstack(Q)).flatten()

    # !! Lake discharge BC
    dQds[0] = (Q[0] - Q_lake)/dx

    # Compute components of time derivatives, disallowing freeze-on and
    # creep opening
    mdot = Q/Lf*(gamma - 1)*dpwds
    mdot[mdot<0] = 0

    creep_closure = 2*S*A*(p_i - pw)**n/n**n
    creep_closure[creep_closure<0] = 0

    # Compute time derivatives (area, pressure, and lake level)
    dSdt = mdot/rhoi - creep_closure
    dpwdt = -1/beta/S*(dSdt + dQds - mdot/rhow)

    if isinstance(A_lake, float):
        A_i = A_lake
    else:
        A_i = A_lake(h)
    dhdt = -Q_lake/A_i

    # Populate state vector
    vprime = np.zeros(2*N+1)
    vprime[:N] = dSdt
    vprime[N:2*N] = dpwdt
    vprime[-1] = dhdt

    return vprime

def solve_compressible(params):
    """
    Solve compressible conduit equations for experimental design
    specified by `params`
    """
    x = params['x']
    L = params['L']
    N = params['N']
    dx = params['dx']

    zb = params['zb']
    p_i = params['p_i']

    dt = params['dt']
    t_eval = params['t_eval']
    t_span = params['t_span']

    A_lake = params['A_lake']

    S = params['init_S']
    pw = params['init_pw']
    h_lake = params['init_h_lake']

    fR = params['fR']

    # DERIVATIVE OPERATORS
    # Upwind derivatives - this enforces a BC on the left boundary
    D_upwind = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
    D_upwind = D_upwind/dx

    # Downwind derivatives - this enforces a BC on the right boundary
    D_downwind = 1/dx*(-np.diag(np.ones(N)) + np.diag(np.ones(N-1), k=1))

    v0 = wrap(S, pw, h_lake)

    fun = lambda t, y: rhs_compressible_channel(t, y, params)
    sol_out = scipy.integrate.solve_ivp(fun, t_span,
        v0, t_eval=t_eval, method='BDF')

    v_sol = sol_out.y
    S = v_sol[:N]
    pw = v_sol[N:2*N]
    h_lake = v_sol[-1]

    # Calculate Q at each timestep
    Q = np.zeros(S.shape)
    for i in range(pw.shape[1]):
        dpwds = np.matmul(D_downwind, np.vstack(pw[:, i])).flatten()
        Q[:, i] = -S[:, i]**(5/4)*(1/rhow/fR)**(1/2)*(1/np.pi)**(1/4)*np.abs(dpwds)**(-0.5)*dpwds

    return (S, pw, Q, h_lake)



def solve_incompressible(params):
    """
    Solve incompressible conduit equations with NO pressure coupling
    """
    A = params['A']
    n = params['n']
    fR = params['fR']

    N = params['N']
    dx = params['dx']
    # Finite-difference discretizations

    # Upwind derivatives - this enforces a BC on the left boundary
    D_upwind = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
    D_upwind = D_upwind/dx

    # Downwind derivatives - this enforces a BC on the right boundary
    D_downwind = 1/dx*(-np.diag(np.ones(N)) + np.diag(np.ones(N-1), k=1))

    p_i = params['p_i']

    pw = params['init_pw']
    dpwds = np.matmul(D_downwind, np.vstack(pw)).flatten()

    S = params['init_S']

    maxiter = 50
    tol = 1e-3
    S_tol = 1e-2/86400

    h_lake = params['init_h_lake']

    A_lake = params['A_lake']

    t, t_end = params['t_span']
    dt = params['dt']
    m = len(params['t_eval'])

    S_out = np.zeros((N, m))
    Q_out = np.zeros((N, m))
    pw_out = np.zeros((N, m))
    h_lake_out = np.zeros(m)

    t = 0

    Q_forcing_handle = params['Q_lake']
    for tindex, t in enumerate(params['t_eval']):
        err = 1e3
        itnum = 0

        p_lake = rhow*g*h_lake
        # Compute pressure gradient at lake outlet
        pgrad_lake = (pw[0] - p_lake)/dx

        if params['drainage']=='pressure-coupled':
            # Compute Q discharge from lake (avoiding divide by zero error)
            if pgrad_lake!=0:
                Q_lake = -S[0]**(5/4)*(1/rhow/fR)**(1/2)*(1/np.pi)**(1/4)*np.abs(pgrad_lake)**(-0.5)*pgrad_lake
            else:
                Q_lake = 0

            # Do not allow negative discharge (e.g. lake filling from conduit)
            if Q_lake<0:
                Q_lake = 0

        else:
            Q_lake_fun = params['Q_lake']
            Q_lake = Q_lake_fun(t)

        # Stop draining when h is <= 0
        if h_lake<=0:
            Q_lake = 0

        # Iterate to find a consistent solution for flux Q and cross sectional
        # area, given an imposed pressure gradient
        while err>tol and itnum<maxiter:
        # for i in range(5):
            # Solve for discharge Q
            D = D_upwind + (1/rhoi - 1/rhow)/Lf*(gamma - 1)*np.diag(dpwds)
            y = np.vstack(2*S*A*( p_i - pw)**n/n**n)
            y[y<0] = 0
            y[0] = y[0] + Q_lake/dx     # Boundary condition on Q[0]

            Q = np.linalg.solve(D, y).flatten()

            # Now calculate pressure gradient - note the negative sign!
            # dpwds = -(Q*(np.pi**(1/4))*(fR*rhow)**(1/2)/(S**(5/4)))**2
            dpwds = -Q**2 * np.pi**(1/2) * (fR*rhow) / (S**(5/2))

            # Integrate to calculate pressure
            yp = np.vstack(dpwds)
            p_new = np.linalg.solve(D_downwind, yp).flatten()

            err = np.max(np.abs(pw - p_new))
            pw = p_new
            itnum += 1


        D = D_upwind + (1/rhoi - 1/rhow)/Lf*(gamma - 1)*np.diag(dpwds)
        y = np.vstack(2*S*A*( (p_i - pw)/n)**n)
        y[y<0] = 0
        y[0] = y[0] + Q_lake/dx     # Boundary condition on Q[0]

        Q = np.linalg.solve(D, y).flatten()

        mi = Q/Lf*(gamma-1)*dpwds
        mi[mi<0] = 0

        creep = 2*S*A*(p_i-pw)**n / n**n
        creep[creep<0] = 0

        dSdt = mi/rhow - creep
        t = t + dt

        if isinstance(A_lake, float):
            A_i = A_lake
        else:
            A_i = A_lake(h_lake)

        h_lake = h_lake - dt*Q_lake/A_i
        if h_lake<0:
            h_lake = 0

        S_err = np.max(np.abs(dSdt))

        S_out[:, tindex] = S
        Q_out[:, tindex] = Q
        pw_out[:, tindex] = pw
        h_lake_out[tindex] = h_lake

        S = S + dt*dSdt


    return S_out, pw_out, Q_out, h_lake_out
