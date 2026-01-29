#%%
import os
import numpy as np
import matplotlib.pyplot as plt

def load_beam_txt(path):
	# Returns ell, Bl, ferr
	arr = []
	with open(path, 'r') as f:
		for line in f:
			if line.strip().startswith('#') or not line.strip():
				continue
			arr.append([float(x) for x in line.strip().split()])
	arr = np.array(arr)
	return arr[:,0], arr[:,1], arr[:,2]

def plot_wmap_band_beams(beam_dir, bands=['K','W']):
	plt.figure(figsize=(10,6))
	for band in bands:
		# New beam
		new_path = os.path.join(beam_dir, f'wmap_ampl_bl_{band}_9yr_v5p1.txt')
		ell, bl, ferr = load_beam_txt(new_path)
		# Old beam (no-weighted)
		old_path = os.path.join(beam_dir, f'wmap_ampl_bl_{band}_9yr_v5p1_nw.txt')
		if os.path.exists(old_path):
			ell_old, bl_old, ferr_old = load_beam_txt(old_path)
		else:
			ell_old, bl_old, ferr_old = None, None, None

		# Plot T (Bl)
		plt.subplot(2,2,1 if band=='K' else 2)
		plt.title(f'{band} band: Temperature')
		plt.plot(ell, bl, label='New (weighted)')
		if bl_old is not None:
			plt.plot(ell_old, bl_old, '--', label='Old (no-weight)')
		plt.xlabel(r'$\ell$')
		plt.ylabel(r'$B_\ell$')
		plt.legend()

		# Plot P (Bl * 0.99 as proxy, since WMAP beams are nearly identical for T/P)
		plt.subplot(2,2,3 if band=='K' else 4)
		plt.title(f'{band} band: Polarization')
		plt.plot(ell, bl, label='New (weighted)')
		if bl_old is not None:
			plt.plot(ell_old, bl_old, '--', label='Old (no-weight)')
		plt.xlabel(r'$\ell$')
		plt.ylabel(r'$B_\ell$ (P)')
		plt.legend()

	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	# Directory containing the beam txt files
	import sys
	if len(sys.argv) > 1:
		beam_dir = '/media/pablo/cmb_ssd/maps/WMAP/beam/'
	else:
		beam_dir = os.path.dirname(os.path.abspath(__file__))
	plot_wmap_band_beams(beam_dir, bands=['K','W'])
