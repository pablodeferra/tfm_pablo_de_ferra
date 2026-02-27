#%%
import numpy as np
from astropy.io import fits
import sys
sys.path.append('../') 
import functions as functions
from data import data, masks
import matplotlib.pyplot as plt
import healpy as hp

mask_alberto_path = '/home/pablo/Downloads/mask_galcut10_dec_6_70_apodC2_5.fits'
mask_alberto = hp.read_map(mask_alberto_path)

mask_path = masks['QUIJOTE_galcut']['galcut10']['path']
mask = hp.read_map(mask_path)

hp.mollview(mask_alberto)
hp.mollview(mask)
hp.mollview(mask - mask_alberto, norm='hist')
plt.savefig('/home/pablo/Desktop/master/tfm/figures/masks/mask_difference.pdf')

#%%

hmdm_path = data['QUIJOTE']['11']['hmdm']

hmdm = hp.read_map(hmdm_path, field=[0,1,2])


def prepare_noise_map(path,txtfreq):
    comp = "IQU"
    ff1  = path+'quijote_mfi_skymap_'+txtfreq+'ghz_512_dr1_half1.fits'
    ff2  = path+'quijote_mfi_skymap_'+txtfreq+'ghz_512_dr1_half2.fits'
    h1   = hp.read_map(ff1,[c + "_STOKES" for c in comp],nest=False)
    h2   = hp.read_map(ff2,[c + "_STOKES" for c in comp],nest=False)

    w1  = hp.read_map(ff1,["WEI_"+c for c in comp],nest=False)
    w2  = hp.read_map(ff2,["WEI_"+c for c in comp],nest=False)
    w1[np.isnan(w1)]=0
    w2[np.isnan(w2)]=0
    w1[w1<0]=0  # Healpy bad values
    w2[w2<0]=0  
    
    w   = np.sqrt( (w1+w2)*(1./w1 + 1./w2) )
    n   = (h1-h2)/w
    n[w1*w2==0]=0
    return(n)

path_map = '/media/pablo/cmb_ssd/maps/QUIJOTE/hmdm/'

# noise_11 = prepare_noise_map(path_map, '11')
noise_11 = hp.read_map('/home/pablo/Downloads/noise11.fits', field=[0,1,2])

comp = ['I', 'Q', 'U']

for ii in range(3):
    print(f'=============== {ii} ===============')
    # hp.mollview(hmdm[ii], norm='hist')
    # hp.mollview(noise_11[ii], norm='hist')
    hp.mollview(hmdm[ii] - noise_11[ii], norm='hist', title=f'{comp[ii]}: pablo - alberto')
    plt.savefig(f'/home/pablo/Desktop/master/tfm/figures/maps/HMDM_difference_{comp[ii]}')
#%%

import pymaster as nmt



# The function defined below will compute the power spectrum between two
# NmtFields f_a and f_b, using the coupling matrix stored in the
# NmtWorkspace wsp and subtracting the deprojection bias clb.
# Note that the most expensive operations in the MASTER algorithm are
# the computation of the coupling matrix and the deprojection bias. Since
# these two objects are precomputed, this function should be pretty fast!
def compute_master(f_a, f_b, wsp):#, clb):
    # Compute the power spectrum (a la anafast) of the masked fields
    # Note that we only use n_iter=0 here to speed up the computation,
    # but the default value of 3 is recommended in general.
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    # Decouple power spectrum into bandpowers inverting the coupling matrix
    cl_decoupled = wsp.decouple_cell(cl_coupled)#, cl_bias=clb)

    return cl_decoupled

def step1_compute_workspace(mask, map, lmax, dl, purify_e=False, purify_b=False, beam=None):

	#Create field spin0
	print(mask.shape)
	print(map[0,:].shape)
	f0 = nmt.NmtField(mask, [map[0,:]], lmax_sht=lmax, beam=beam)
	#Create field spin2
	f2 = nmt.NmtField(mask, [map[1,:],map[2,:]], lmax_sht=lmax, purify_e=purify_e, purify_b=purify_b, beam=beam)
                
	#Create binning scheme
	dl = 20
	nside = hp.npix2nside(np.size(mask))
	b = nmt.NmtBin(nside, nlb=dl, lmax=lmax)
	n_ell = b.get_n_bands()

	#Generate Workspace object, that is based only on the masks of the objects to correlate (This computes the coupling matrix that will be applied to all the power spectra)
	w00 = nmt.NmtWorkspace()
	w00.compute_coupling_matrix(f0, f0, b)

	w02 = nmt.NmtWorkspace()
	w02.compute_coupling_matrix(f0, f2, b)

	w22 = nmt.NmtWorkspace()
	w22.compute_coupling_matrix(f2, f2, b)

	return(f0, f2, b, w00, w02, w22)

def step2_compute_spectrum(mask, map, f0, f2, b, w00, w02, w22):
	#Create field spin0
	#f0 = nmt.NmtField(mask, [map[0,:]])
	#Create field spin2
	#f2 = nmt.NmtField(mask, [map[1,:],map[2,:]])

	# OK, we can now compute the power spectrum of our two input fields
	cl_master_tt = compute_master(f0, f0, w00) # TT
	cl_master_tetb = compute_master(f0, f2, w02) # TE TB
	cl_master_eb = compute_master(f2, f2, w22) # EE EB BE BB
            
	#plot power spectra
	cl_tt =  cl_master_tt[0] #label='TT '
	cl_te = cl_master_tetb[0] #label='TE '
	cl_tb = cl_master_tetb[1] #label='TB '
	cl_ee = cl_master_eb[0] #label='EE '
	cl_eb = cl_master_eb[1] #label='EB '
	cl_be = cl_master_eb[2] #label='BE '
    
	cl_bb = cl_master_eb[3] #label='BB ' 

	out = np.array([b.get_effective_ells(), cl_tt, cl_ee, cl_bb, cl_te, cl_tb, cl_eb, cl_be])

	return(out)

lmax = 2*512+1
dl=20


f010, f210, b10, w0010, w0210, w2210 = step1_compute_workspace(mask_alberto, noise_11, lmax=lmax, dl=dl, purify_e=True, purify_b=True, beam=None)

cl_alberto_10 = step2_compute_spectrum(mask_alberto, noise_11, f010, f210, b10, w0010, w0210, w2210)

ell_alb = cl_alberto_10[0]
nl_EE_alb = cl_alberto_10[2]
nl_BB_alb = cl_alberto_10[3]

binning_params_lin = {
    'type': 'linear',
    'lmax': lmax,
    'dl': dl}

b_lin = functions.create_binning(binning_params_lin)
workspaces_lin = functions.prepare_workspaces(mask, b_lin, 512, lmax=lmax, purify_e=True, purify_b=True)



hmdm_lin = functions.compute_hmdm_power_spectra(
    data, ['11'], mask, b_lin,
    workspaces=workspaces_lin,
    lmax=lmax,
    use_noise=False
)

ell_pab = hmdm_lin[0, 0]['ell_eff']
nl_EE_pab = hmdm_lin[0, 0]['EE']
nl_BB_pab = hmdm_lin[0, 0]['BB']

alberto_path = '/home/pablo/Downloads/spectra311_galcut10_noise_Nmt_pure.txt'

alberto_nl = np.loadtxt(alberto_path, skiprows=1).T

ell_alb_2 = alberto_nl[0]
nl_EE_alb_2 = alberto_nl[2]
nl_BB_alb_2 = alberto_nl[3]

#%%

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- EE ----
ax = axes[0]
ax.plot(ell_pab, nl_EE_pab, '-', label=r'$N_\ell^{EE}$ (Pablo)', color='k')
ax.plot(ell_alb, nl_EE_alb, '--', label=r'$N_\ell^{EE}$ (Alberto)', color='steelblue')
ax.plot(ell_alb_2, nl_EE_alb_2, '-.', label=r'$N_\ell^{EE}$ (spectra311)', color='purple')
# ax.plot(ell_alb, nl_EE_pab - nl_EE_alb + 1e-5, ':', label=r'1e-5 + $N_\ell^{EE}$ (Pablo) - $N_\ell^{EE}$ (Alberto)', color='goldenrod')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [m K^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — EE', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(0, 200)

# ---- BB ----
ax = axes[1]
ax.plot(ell_pab, nl_BB_pab, '-', label=r'$N_\ell^{BB}$ (Pablo)', color='k')
ax.plot(ell_alb, nl_BB_alb, '--', label=r'$N_\ell^{BB}$ (Alberto)', color='steelblue')
ax.plot(ell_alb_2, nl_BB_alb_2, '-.', label=r'$N_\ell^{BB}$ (spectra311)', color='purple')
# ax.plot(ell_alb, nl_BB_pab - nl_BB_alb + 1e-5, ':', label=r'1e-5 + $N_\ell^{BB}$ (Pablo) - $N_\ell^{BB}$ (Alberto)', color='goldenrod')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', fontsize=13)
ax.set_ylabel(r'$C_\ell \; [m K^2]$', fontsize=13)
ax.set_title('QUIJOTE 11 GHz — BB', fontsize=13)
ax.legend(frameon=False)
ax.set_xlim(0, 200)

plt.tight_layout()
plt.show()

fig.savefig('/home/pablo/Desktop/master/tfm/figures/spectra/Nl_HMDM_code_pablo_alberto_short_dl20.pdf')
