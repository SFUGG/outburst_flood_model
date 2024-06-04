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

S_i, pw_i, Q_i, h_i = models.solve_incompressible(params)


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

ax1.plot(x, S_i[:, -1], label='Incompressible')

ax2.plot(t_eval/86400, Q_i[0], label='Incompressible')


S_c, pw_c, Q_c, h_c = models.solve_compressible(params)

ax1.plot(x, S_c[:, -1], label='Compressible')

ax2.plot(t_eval/86400, Q_c[0], label='Compressible')

ax1.grid()
ax1.set_ylabel('S (m$^2$)')
ax1.set_xlabel('Distance along conduit (km)')
ax1.text(0.1, 0.925, 'a', transform=ax1.transAxes, fontsize=12)

ax2.grid()
ax2.set_ylabel('Q (m$^3$/s)')
ax2.set_xlabel('Time (days)')
ax2.text(0.1, 0.925, 'b', transform=ax2.transAxes, fontsize=12)
# ax2.legend()
ax2.set_ylim([0, 100])

plt.tight_layout()

fig.savefig('pressure_discharge_comparison.png', dpi=600)

fig, axs = plt.subplots(nrows=4, figsize=(7, 8), sharex=True)
axs = axs.flatten()

axs[0].plot(t_eval/86400, h_c)
axs[0].grid()
# axs[0].set_xlabel('Time (days)')
axs[0].set_ylabel('h$_r$ (m)')
axs[0].text(0.025, 0.85, 'a', transform=axs[0].transAxes, fontsize=12)

axs[1].plot(t_eval/86400, Q_c[0])
axs[1].grid()
# axs[1].set_xlabel('Time (days)')
axs[1].set_ylabel('Q$_r$ (m)')
axs[1].text(0.025, 0.85, 'b', transform=axs[1].transAxes, fontsize=12)


axs[2].plot(t_eval/86400, pw_c[0]/p_i[0])
axs[2].grid()
# axs[2].set_xlabel('Time (days)')
axs[2].set_ylabel('p$_r$/p$_i$')
axs[2].text(0.025, 0.85, 'c', transform=axs[2].transAxes, fontsize=12)


axs[3].plot(t_eval/86400, S_c[0])
# axs[3].set_xlabel('Time (days)')
axs[3].grid()
axs[3].set_ylabel('Outlet area (m$^2$)')
axs[3].text(0.025, 0.85, 'd', transform=axs[3].transAxes, fontsize=12)


plt.tight_layout()
fig.savefig('compressible_pressure_drainage.png', dpi=600)

plt.show()
