"""
Imposed discharge for incompressible and compressible channel models
"""

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

params_const = params.copy()
params_const['A_lake'] = lambda h: A_lake*(h+5)/(h+5)

params_hyps = params.copy()


tot_vol = A_lake*h_lake

base_area = 3*tot_vol/h_lake

slope = h_lake/np.sqrt(base_area)

area = lambda h: (h/slope)**2
params_hyps['A_lake'] = lambda h: area(h) + 10

hh = np.arange(0, h_lake)
# fig, ax = plt.subplots()
# ax.plot(hh, params_const['A_lake'](hh))
# ax.plot(hh, params_hyps['A_lake'](hh))
# plt.show()

dh = hh[1:] - hh[:-1]

res_const = models.solve_compressible(params_const)
res_hyps = models.solve_compressible(params_hyps)


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

ax1.plot(t_eval/86400, res_const[2][0], label='Constant')
ax1.plot(t_eval/86400, res_hyps[2][0], label='Hypsometric')
ax1.grid()
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Discharge (m$^3$/s)')
ax1.text(0.025, 0.95, 'a', transform=ax1.transAxes, fontsize=12)
ax1.set_xlim([0, 60])
ax1.set_ylim([0, 300])
# ax1.legend()

ax2.plot(t_eval/86400, res_const[-1], label='Constant')
ax2.plot(t_eval/86400, res_hyps[-1], label='Hypsometric')
ax2.grid()
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('h (m)')
ax2.text(0.025, 0.95, 'b', transform=ax2.transAxes, fontsize=12)
ax2.set_xlim([0, 60])
ax2.legend(loc='left')
ax2.set_ylim([0, 550])

plt.tight_layout()
# ax1.legend()

fig.savefig('hypsometry_discharge.png', dpi=600, fontsize=12)


plt.show()
