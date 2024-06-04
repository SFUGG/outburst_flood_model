"""
Sensitivity to compressibility beta and friction fR for pressure-coupled model
"""

import os

import numpy as np
from matplotlib import pyplot as plt

import models

# CONSTANTS needed for defining model specification
rhow = 1000
g = 9.81

# DOMAIN
L = 10e3            # Domain length (m)
N = 100             # Number of grid cells
dx = (L/N)          # Derived grid spacing
x = np.arange(0, L, dx)

bed_slope = 0               # Bed slope (-)
zb = 100 - bed_slope*x      # Bed elevation (m)

# Compute suface elevation zs (m)
zs_arg = L - x + 2*dx
zs_arg[zs_arg<0] = 0
zs = 600/np.sqrt(L)*np.sqrt(zs_arg)
zs[zs<=zb] = zb[zs<=zb]
p_i = rhow*g*(zs - zb)      # Overburden ice pressure (Pa)

params = models.default_params.copy()
params['L'] = L
params['N'] = N
params['dx'] = dx
params['x'] = x

params['zb'] = zb
params['p_i'] = p_i

# Timestepping
dt = 3600
t = 0
t_end = 86400*60
t_span = (t, t_end)
t_eval = np.arange(t, t_end, dt)

params['dt'] = dt
params['t_span'] = t_span
params['t_eval'] = t_eval

h_lake = p_i[0]/rhow/g
p_lake = rhow*g*h_lake
A_lake = 500*500

params['A_lake'] = A_lake
params['init_h_lake'] = h_lake

# PRESCRIBED LAKE DRAINAGE
params['Q_lake'] = lambda t: 20
params['drainage'] = 'pressure-coupled'

params['beta'] = 1e-7

# Initial pressure guess - needed in first iteration to solve for
# first guess of discharge Q. This is not an initial condition but a "first
# guess" in the iteration step
pw = p_i
params['init_pw'] = pw

# INITIAL CONDITIONS
S = 0.1 + 0*x/L   # True initial condition on channel cross-section

params['init_S'] = S

fR_vals = [0.05, 0.1, 0.15, 0.2, 0.25]
n_t = len(t_eval)
n_fR = len(fR_vals)
Q_fR = np.zeros((n_t, n_fR))

beta_vals = [1e-7, 1e-6, 1e-5, 1e-4]
n_beta = len(beta_vals)
Q_beta = np.zeros((n_t, n_beta))

if not os.path.exists('data/Q_fR.npy'):
    for (i, fR) in enumerate(fR_vals):
        print(fR)
        sens_params = params.copy()
        sens_params['fR'] = fR

        S, pw, Q, h = models.solve_compressible(sens_params)
        Q_fR[:, i] = Q[0]
    np.save('data/Q_fR.npy', Q_fR)   

if not os.path.exists('data/Q_beta.npy'):
    for (j, beta) in enumerate(beta_vals):
        print(beta)
        sens_params = params.copy()
        sens_params['beta'] = beta

        S, pw, Q, h = models.solve_compressible(sens_params)
        Q_beta[:, j] = Q[0]
    np.save('data/Q_beta.npy', Q_beta)

Q_fR = np.load('data/Q_fR.npy')
Q_beta = np.load('data/Q_beta.npy')
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)
for (i, fR) in enumerate(fR_vals):
    ax1.plot(t_eval/86400, Q_fR[:, i], label='f$_R$ = %.2f' % fR)

for (j, beta) in enumerate(beta_vals):
    ax2.plot(t_eval/86400, Q_beta[:, j], label='$\\beta$ = %.1e' % beta)


ax1.grid()
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Discharge (m$^3$/s)')
ax1.text(0.025, 0.95, 'a', transform=ax1.transAxes, fontsize=12)
ax1.legend(loc='upper right')
ax1.set_xlim([0, 60])
ax1.set_ylim([0, 225])

ax2.grid()
ax2.set_xlabel('Time (days)')
ax2.text(0.025, 0.95, 'b', transform=ax2.transAxes, fontsize=12)
ax2.legend()
ax2.set_xlim([0, 60])

fig.savefig('figures/sensitivity_tests.png', dpi=600)
