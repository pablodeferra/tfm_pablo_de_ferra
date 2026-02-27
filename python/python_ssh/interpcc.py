# 2021-03-9
# 2021-03-10 CLC	I start a new code to compute all cases.
# 2021-03-11  MP   Reformat into functions
# 2021-03-16  MP   Add IRAS, Planck HFIs
# 2021-03-18  MP   If using Planck, halve the array to reduce CPU time
# 2021-03-19  MP   Add regular grid interpolation to speed things up

import numpy as np
from astropy.io import fits
from scipy import interpolate

def interpcc_setup(infile,band,td_limit=40,method=2):
	# Read in the fits file with precomputed values
	dat = fits.open(infile)
	# print(dat.info())
	# print(dat[1].header)
	bands = dat[1].data[0][0]
	td = dat[1].data[0][1]
	beta = dat[1].data[0][2]
	#
	doing_planck = False
	if band.startswith('DB'):
		band = band.split('DB')[1]
	elif band.startswith('P'):
		band = band.split('P')[1]
		doing_planck = True
	elif band.startswith('I'):
		iras_bands = {'I100': '4', 'I60': '3', 'I25': '2', 'I12': '1'}
		band = iras_bands.get(band, 0)
	idx_band = [ii for ii,bb in enumerate(bands) if band ==bb]

	if len(idx_band) == 0:
		raise ValueError(f"Invalid band name '{band}'")

	if doing_planck:
		map_cc = dat[1].data[0][3][1][idx_band[0]]
		# map_cc = map_cc[::2,::2]
		# td = td[::2]
		# beta = beta[::2]
	else:
		map_cc = dat[1].data[0][3][idx_band[0]]

	# Limit the dust tempertaure
	sel_td = (td <= td_limit)
	X, Y = np.meshgrid(beta,td[sel_td])
	Z = map_cc[:,sel_td].T

	# Interpolation
	if method == 1:
		# Method 1: using interp2d
		return interpolate.interp2d(X, Y, Z, kind ='cubic')
	elif method == 2:
		# Method 2: using Rbf
		return interpolate.Rbf(X,Y,Z,function='cubic')
	else:
		# Method 3, using RegularGridInterpolator
		return interpolate.RegularGridInterpolator((td[sel_td],beta),Z,method='linear')

def interpcc(interp,td,bd):
	try:
		return np.around(interp(bd,td)[()],4)
	except:
		return np.around(interp([td,bd])[()],4)[0]