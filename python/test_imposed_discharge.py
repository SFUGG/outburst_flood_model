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
zb = 0 - bed_slope*x      # Bed elevation (m)

# Compute suface elevation zs (m)
zs_arg = L - x
# zs_arg[zs_arg<=0] = 1
zs = 600/np.sqrt(L)*np.sqrt(zs_arg)
zs[zs<=zb] = zb[zs<=zb]
p_i = rhow*g*(zs - zb)      # Overburden ice pressure (Pa)

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(x/1e3, zb, 'k')
ax.plot(x/1e3, zs, 'b')
ax.set_xlabel('Distance along conduit (km)')
ax.set_ylabel('Elevation (m)')
ax.grid()
plt.tight_layout()
fig.savefig('domain.png', dpi=600)

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
t_end = 86400*15
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
params['drainage'] = 'imposed'

params['beta'] = 1e-7

# Initial pressure guess - needed in first iteration to solve for
# first guess of discharge Q. This is not an initial condition but a "first
# guess" in the iteration step
pw = p_i
params['init_pw'] = pw

# INITIAL CONDITIONS
S = 1 + 0*x/L   # True initial condition on channel cross-section

params['init_S'] = S

S_i, pw_i, Q_i, h_i = models.solve_incompressible(params)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

ax1.plot(x/1e3, S_i[:, -1], label='Incompressible')

ax2.plot(t_eval/86400, S_i[0], label='Incompressible')


S_c, pw_c, Q_c, h_c = models.solve_compressible(params)

ax1.plot(x/1e3, S_c[:, -1], label='Compressible')

ax2.plot(t_eval/86400, S_c[0], label='Compressible')

ax1.grid()
ax1.set_ylabel('S (m$^2$)')
ax1.set_xlabel('Distance along conduit (km)')
ax1.text(0.1, 0.925, 'a', transform=ax1.transAxes, fontsize=12)

ax2.grid()
ax2.set_ylabel('S (m$^2$)')
ax2.set_xlabel('Time (days)')
ax2.text(0.1, 0.925, 'b', transform=ax2.transAxes, fontsize=12)
ax2.legend()

plt.tight_layout()

fig.savefig('imposed_discharge_comparison.png', dpi=600)

