#%%
import numpy as np
import healpy as hp

# Read half-mission maps and prepares noise map
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

noise_11 = prepare_noise_map('/home/pablo/Desktop/Fisica/TFG/maps/', '11')
noise_13 = prepare_noise_map('/home/pablo/Desktop/Fisica/TFG/maps/', '13')
noise_17 = prepare_noise_map('/home/pablo/Desktop/Fisica/TFG/maps/', '17')
noise_19 = prepare_noise_map('/home/pablo/Desktop/Fisica/TFG/maps/', '19')

hp.write_map('/home/pablo/Desktop/Fisica/TFG/maps/quijote_311_HMDM.fits', noise_11)
hp.write_map('/home/pablo/Desktop/Fisica/TFG/maps/quijote_313_HMDM.fits', noise_13)
hp.write_map('/home/pablo/Desktop/Fisica/TFG/maps/quijote_417_HMDM.fits', noise_17)
hp.write_map('/home/pablo/Desktop/Fisica/TFG/maps/quijote_419_HMDM.fits', noise_19)


#%%
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm

half_1, h1 = hp.read_map('/home/pablo/Desktop/Fisica/TFG/maps/PR3/LFI_SkyMap_030-BPassCorrected_1024_R3.00_full-ringhalf-1.fits', field=[0,1,2], h=True)
half_2, h2 = hp.read_map('/home/pablo/Desktop/Fisica/TFG/maps/PR3/LFI_SkyMap_030-BPassCorrected_1024_R3.00_full-ringhalf-2.fits', field=[0,1,2], h=True)

sigma_1 = np.sqrt(hp.read_map('/home/pablo/Desktop/Fisica/TFG/maps/PR3/LFI_SkyMap_030-BPassCorrected_1024_R3.00_full-ringhalf-1.fits', field=[4,7,9]))
sigma_2 = np.sqrt(hp.read_map('/home/pablo/Desktop/Fisica/TFG/maps/PR3/LFI_SkyMap_030-BPassCorrected_1024_R3.00_full-ringhalf-2.fits', field=[4,7,9]))
w = np.sqrt((1/sigma_1**2 + 1/sigma_2**2)*(sigma_1**2 + sigma_2**2))

nside = 1024
npix = hp.nside2npix(nside) 

hmdm = 1e3* (half_1 - half_2) / w 

hp.write_map('/home/pablo/Desktop/Fisica/TFG/maps/PR3/planck_030_HMDM.fits', hmdm, overwrite=True)

#%%
import numpy as np
import healpy as hp

mapa1 = hp.read_map('/home/pablo/Desktop/Paper/maps/WMAP/hmdm/wmap_iqumap_r9_yr1to4_K_v5.fits')
mapa2 = hp.read_map('/home/pablo/Desktop/Paper/maps/WMAP/hmdm/wmap_iqumap_r9_yr5to9_K_v5.fits')

hd = (mapa1-mapa2)*np.sqrt(20/81)

#%%

from data import data, path_map, masks, path_masks

