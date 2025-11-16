#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, h, k
import sys
sys.path.append('../') 
from functions import cmb_unit_conversion, load_cmb_spectrum_from_file

# Path to Planck CMB spectrum file
cmb_spectrum_file = "/home/pablo/Desktop/master/tfm/spectra/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"

# Multipole range
ell_values = np.arange(2, 200)

# Load CMB spectrum (returns C_l in K²)
print("[INFO] Loading CMB spectrum from Planck file...")
cmb_spectra = load_cmb_spectrum_from_file(cmb_spectrum_file, ell_values, planck_format=True)

# Get EE component
Cl_EE_K2 = cmb_spectra['EE']

# Convert to mK_CMB²
Cl_EE_mK2_CMB = Cl_EE_K2 * 1e6

# Convert to different frequencies in K_RJ²
frequencies = [11.1, 13.0, 17.0, 19.0, 23.0, 30.0, 44.0, 70.0, 100.0, 143.0, 217.0, 353.0]

# Create plot
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot 1: CMB EE in mK_CMB²
ax1 = axes[0]
ax1.plot(ell_values, Cl_EE_mK2_CMB, 'k-', lw=2, label='CMB EE')
ax1.set_xlabel(r'Multipole $\ell$', fontsize=12)
ax1.set_ylabel(r'$C_\ell^{EE}$ [mK$_{\rm CMB}^2$]', fontsize=12)
ax1.set_title('CMB EE Spectrum in Thermodynamic Temperature', fontsize=14)
ax1.set_xlim(2, 200)
ax1.set_yscale('log')
ax1.legend(fontsize=10, frameon=False)

# Plot 2: CMB EE in K_RJ² for different frequencies
ax2 = axes[1]

colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))

for i, freq in enumerate(frequencies):
    # Get conversion factor K_CMB → K_RJ for this frequency
    uc_factor = cmb_unit_conversion(freq, 'KCMB2KRJ')
    
    # Convert spectrum: K² → K_RJ²
    Cl_EE_KRJ2 = Cl_EE_K2 * uc_factor**2
    
    # Convert to mK_RJ²
    Cl_EE_mK2_RJ = Cl_EE_KRJ2 * 1e6
    
    ax2.plot(ell_values, Cl_EE_mK2_RJ, lw=1.5, label=f'{freq} GHz', color=colors[i])

ax2.set_xlabel(r'Multipole $\ell$', fontsize=12)
ax2.set_ylabel(r'$C_\ell^{EE}$ [mK$_{\rm RJ}^2$]', fontsize=12)
ax2.set_title('CMB EE Spectrum in Rayleigh-Jeans Temperature (frequency dependent)', fontsize=14)
ax2.set_xlim(2, 200)
ax2.set_yscale('log')
ax2.legend(fontsize=8, ncol=2, loc='upper right', frameon=False)

plt.tight_layout()
plt.show()


