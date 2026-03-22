#%%
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.labelcolor': 'white',     # Axis labels
    'xtick.color': 'white',         # X-axis tick labels
    'ytick.color': 'white',         # Y-axis tick labels
    'axes.titlecolor': 'white',     # Title color
    'legend.facecolor': 'black',    # Legend background
    'legend.edgecolor': 'white',    # Legend border
    'legend.fontsize': 'medium',    # Legend font size
    'text.color': 'white',          # General text color
    'figure.facecolor': 'black',    # Figure background
    'figure.edgecolor': 'white',    # Figure edge color
    'axes.facecolor': 'black',      # Axes background
    'axes.edgecolor': 'white'       # Axes edge (border) color
})


# Frequencies in GHz
nu = np.logspace(np.log10(10), np.log10(1000), 500)

# Physical constants
h = 6.62607015e-34
k_B = 1.380649e-23
T_cmb = 2.72548

def x_calc(nu_GHz):
    return (h * nu_GHz * 1e9) / (k_B * T_cmb)

def cmb_thermo_to_rj(nu_GHz):
    x = x_calc(nu_GHz)
    return (x**2 * np.exp(x)) / (np.exp(x) - 1)**2

# Approximate parameters based on the paper figure (polarization)
A_cmb = 0.55  # muK_RJ at low freqs roughly
CMB = A_cmb * cmb_thermo_to_rj(nu)

# Synchrotron: nu^\beta_s
A_sync = 3.3
beta_s = -3.1
Sync = A_sync * (nu / 30.0)**beta_s

# Thermal DustS
def dust_sed(nu_GHz, A_d_353, beta_d=1.5, T_d=19.6):
    nu_0 = 353.0
    num = np.exp(h * nu_0 * 1e9 / (k_B * T_d)) - 1
    den = np.exp(h * nu_GHz * 1e9 / (k_B * T_d)) - 1
    return A_d_353 * (nu_GHz / nu_0)**(beta_d + 1) * (num / den)

A_dust = 3.5  # middle of the band
Dust_mid = dust_sed(nu, A_dust)
Dust_high = dust_sed(nu, 5.2) # upper bound
Dust_low = dust_sed(nu, 2.5)  # lower bound

Sync_high = 1.5 * Sync
Sync_low = 0.67 * Sync

Sum_mid = CMB + Sync + Dust_mid
Sum_high = CMB + Sync_high + Dust_high
Sum_low = CMB + Sync_low + Dust_low

# Sum_mid = CMB + Sync 
# Sum_high = CMB + Sync_high 
# Sum_low = CMB + Sync_low 

fig, ax = plt.subplots(figsize=(10, 7))



# Sum
ax.plot(nu, Sum_mid, color='w', linestyle='--', linewidth=2, label='Sum foregrounds')
ax.plot(nu, Sum_high, color='w', linestyle='--', linewidth=1.5)
ax.plot(nu, Sum_low, color='w', linestyle='--', linewidth=1.5)

# Text labels
ax.text(12, 11, 'Synchrotron', color='steelblue', fontsize=20, rotation=-43)
ax.text(20, 0.6, 'CMB', color='forestgreen', fontsize=20)
# ax.text(220, 4, 'Thermal dust', color='firebrick', fontsize=20, rotation=14)
ax.text(44, 2.5, 'Sum foregrounds', color='w', fontsize=18, rotation=0)


# # --- QUIJOTE bands ---
# ax.axvspan(10, 20, color='magenta', alpha=0.5)

# --- WMAP bands ---
# ax.axvspan(23, 94, color='blue', alpha=0.5)

# --- Planck bands ---
# ax.axvspan(30, 70, color='w', alpha=0.5, hatch='//')
# ax.axvspan(100, 353, color='w', alpha=0.5, hatch='//')


# ax.text(14, 1000, 'MFI', color='magenta', fontsize=20, horizontalalignment='center', fontweight='bold')
# ax.text(47, 1000, 'WMAP', color='blue', fontsize=20, horizontalalignment='center', fontweight='bold')
# ax.text(47, 400, 'LFI', color='w', horizontalalignment='center', fontsize=20, fontweight='bold')
# ax.text(200, 1000, 'HFI', color='w', horizontalalignment='center', fontsize=20, fontweight='bold')


# Foreground bands
ax.fill_between(nu, Sync_low, Sync_high, color='steelblue', alpha=1.0)
ax.fill_between(nu, Dust_low, Dust_high, color='firebrick', alpha=1.0)

# CMB
ax.plot(nu, CMB, color='forestgreen', linewidth=4, label='CMB')

# Axis settings
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(10, 1000)
ax.set_ylim(0.03, 3000)

ax.set_xlabel('Frequency [GHz]', fontsize=22)
ax.set_ylabel(r'Rms brightness temperature [$\mu\mathrm{K}_{\mathrm{RJ}}$]', fontsize=22)

# Ticks
ax.set_xticks([10, 30, 100, 300, 1000])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.tick_params(axis='both', which='major', labelsize=16, direction='in', length=8)
ax.tick_params(axis='both', which='minor', direction='in', length=4)

plt.tight_layout()
# plt.savefig('/home/pablo/Desktop/master/tfm/python/1-methodology_tests/figures/galactic_foregrounds_wmap_quijote.pdf', dpi=300)
plt.savefig('/home/pablo/Desktop/master/tfm/figures_ppt/foregrounds/galactic_foregroundspng', dpi=300, transparent=True)
plt.show()

