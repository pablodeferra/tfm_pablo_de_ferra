#%%
import astropy.units as u

# QUIJOTE:

path_map_quijote = '/net/nas/proyectos/quijote/release/'
path_sim_quijote = '/home/pdeferra-ext/data/PYSM'
path_beam_quijote = '/home/pdeferra-ext/data/beams/QUIJOTE/'
path_hmdm_quijote = '/home/pdeferra-ext/data/HMDM/QUIJOTE/'
path_half1_quijote = '/net/nas/proyectos/quijote/release/'
path_half2_quijote = '/net/nas/proyectos/quijote/release/'
path_noise_quijote = '/net/calp-nas/proyectos/quijote2/tfgi/validation/Nov2020/noise_simulations/RecommendedSimulations/RecommendedSimulations_allsets_sm1deg/'
path_white_noise_quijote = ''

# WMAP:

path_map_wmap = '/net/nas/proyectos/cosmology/wmap/WMAP9/'
path_map_wmap_ind_years = '/net/nas/proyectos/cosmology/wmap/WMAP9/mapas_original/temp_and_pol/ind_years/'
path_sim_wmap = '/home/pdeferra-ext/data/PYSM'
path_beam_wmap = '/home/pdeferra-ext/data/beams/WMAP/'
path_hmdm_wmap = '/home/pdeferra-ext/data/HMDM/WMAP/'
path_half1_wmap = ''
path_half2_wmap = ''
path_noise_wmap = '/net/nas/proyectos/cosmology/wmap/WMAP9/noise/noise_simulations/'
path_white_noise_wmap = ''


# Planck LFI:

path_map_planck = '/net/nas/proyectos/cosmology/planck/maps/2018_release/'
path_sim_planck = '/home/pdeferra-ext/data/PYSM'
path_beam_planck = '/home/pdeferra-ext/data/beams/Planck/'
path_hmdm_planck = '/home/pdeferra-ext/data/HMDM/Planck/'
path_half1_planck = '/net/nas/proyectos/cosmology/planck/maps/2018_release/'
path_half2_planck = '/net/nas/proyectos/cosmology/planck/maps/2018_release/'
path_noise_planck = '/net/nas/proyectos/cosmology/planck/noise_sims/simulated_maps/PR3/noise/'
path_white_noise_planck = ''


# Masks

path_masks = '/home/pdeferra-ext/masks/'


# Color corrections

path_cc = '/home/pdeferra-ext/data/cc/'


data = {
    'color_corrections': {
        'path': path_cc,
        'cache_file': path_cc + 'color_corrections_table.fits',
    },
    'QUIJOTE': {
        '11': {
            'freq': 10.98 * u.GHz,
            'fwhm': 55.38 * u.arcmin,
            'noise_I': u.mK,
            'noise_QU': u.mK,
            'path': path_map_quijote + 'quijote_mfi_skymap_11ghz_512_dr1.fits',
            'path_simulated': path_sim_quijote + 'map_QUIJOTE_11_nside512_beamconv.fits',
            'beam': path_beam_quijote + 'rimo_quijote_mfi_beamtf_dr1.fits',
            'hmdm': path_hmdm_quijote + 'quijote_mfi_skymap_11ghz_512_dr1_hmdm.fits',
            'half_1': path_half1_quijote + 'quijote_mfi_skymap_11ghz_512_dr1_half1.fits',
            'half_2': path_half2_quijote + 'quijote_mfi_skymap_11ghz_512_dr1_half2.fits',
            'path_noise_simulations': path_noise_quijote + '',
            'path_white_noise_simulations': path_white_noise_quijote + '',
            'noise_simulation_1': 'quijote_11GHz_horn3_0001_sm1deg.fits',
            'white_noise_simulation_1': 'white_noise_11ghz_0001.fits',
        },
        '13': {
            'freq': 12.89 * u.GHz,
            'fwhm': 55.84 * u.arcmin,
            'noise_I': u.mK,
            'noise_QU': u.mK,
            'path': path_map_quijote + 'quijote_mfi_skymap_13ghz_512_dr1.fits',
            'path_simulated': path_sim_quijote + '',
            'beam': path_beam_quijote + 'rimo_quijote_mfi_beamtf_dr1.fits',
            'hmdm': path_hmdm_quijote + 'quijote_mfi_skymap_13ghz_512_dr1_hmdm.fits',
            'half_1': path_half1_quijote + 'quijote_mfi_skymap_13ghz_512_dr1_half1.fits',
            'half_2': path_half2_quijote + 'quijote_mfi_skymap_13ghz_512_dr1_half2.fits',
            'path_noise_simulations': path_noise_quijote + '',
            'path_white_noise_simulations': path_white_noise_quijote + '',
            'noise_simulation_1': 'quijote_13GHz_horn3_0001_sm1deg.fits',
            'white_noise_simulation_1': 'white_noise_13ghz_0001.fits',
        },
        '17': {
            'freq': 16.85 * u.GHz,
            'fwhm': 38.95 * u.arcmin,
            'noise_I': u.mK,
            'noise_QU': u.mK,
            'path': path_map_quijote + 'quijote_mfi_skymap_17ghz_512_dr1.fits',
            'path_simulated': path_sim_quijote + '',
            'beam': path_beam_quijote + 'rimo_quijote_mfi_beamtf_dr1.fits',
            'hmdm': path_hmdm_quijote + 'quijote_mfi_skymap_17ghz_512_dr1_hmdm.fits',
            'half_1': path_half1_quijote + 'quijote_mfi_skymap_17ghz_512_dr1_half1.fits',
            'half_2': path_half2_quijote + 'quijote_mfi_skymap_17ghz_512_dr1_half2.fits',
            'path_noise_simulations': path_noise_quijote + '',
            'path_white_noise_simulations': path_white_noise_quijote + '',
            'noise_simulation_1': 'quijote_17GHz_hornx_0001_sm1deg.fits',
            'white_noise_simulation_1': 'white_noise_17ghz_0001.fits',
        },
        '19': {
            'freq': 18.85 * u.GHz,
            'fwhm': 40.32 * u.arcmin,
            'noise_I': u.mK,
            'noise_QU': u.mK,
            'path': path_map_quijote + 'quijote_mfi_skymap_19ghz_512_dr1.fits',
            'path_simulated': path_sim_quijote + '',
            'beam': path_beam_quijote + 'rimo_quijote_mfi_beamtf_dr1.fits',
            'hmdm': path_hmdm_quijote + 'quijote_mfi_skymap_19ghz_512_dr1_hmdm.fits',
            'half_1': path_half1_quijote + 'quijote_mfi_skymap_19ghz_512_dr1_half1.fits',
            'half_2': path_half2_quijote + 'quijote_mfi_skymap_19ghz_512_dr1_half2.fits',
            'path_noise_simulations': path_noise_quijote + '',
            'path_white_noise_simulations': path_white_noise_quijote + '',
            'noise_simulation_1': 'quijote_19GHz_hornx_0001_sm1deg.fits',
            'white_noise_simulation_1': 'white_noise_19ghz_0001.fits',
        },
    },
    'WMAP': {
        '23': {
            'freq': 22.72 * u.GHz,
            'fwhm': 52.8 * u.arcmin,
            'noise_I': 1.429 * u.mK,      
            'noise_QU': 1.435 * u.mK,
            'path': path_map_wmap + 'wmap_band_iqumap_r9_9yr_K_v5.fits',
            'path_simulated': path_sim_wmap + 'map_WMAP_23_nside512_beamconv.fits',
            'beam': path_beam_wmap + 'wmap_ampl_bl_K_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + 'wmap_iqumap_r9_hmdm_K_v5.fits',
            'half_1': path_half1_wmap + 'wmap_iqumap_r9_yr1to4_K_v5.fits',
            'half_2': path_half2_wmap + 'wmap_iqumap_r9_yr5to9_K_v5.fits',
            'path_noise_simulations': path_noise_wmap + '23/',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_23ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_23ghz_0001.fits',
        },
        '33': {
            'freq': 32.98 * u.GHz,
            'fwhm': 39.6 * u.arcmin,
            'noise_I': 1.466 * u.mK,
            'noise_QU': 1.472 * u.mK,
            'path': path_map_wmap + 'wmap_band_iqumap_r9_9yr_Ka_v5.fits',
            'path_simulated': path_sim_wmap + 'map_WMAP_33_nside512_beamconv.fits',
            'beam': path_beam_wmap + 'wmap_ampl_bl_Ka_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + 'wmap_iqumap_r9_hmdm_Ka_v5.fits',
            'half_1': path_half1_wmap + 'wmap_iqumap_r9_yr1to4_Ka_v5.fits',
            'half_2': path_half2_wmap + 'wmap_iqumap_r9_yr5to9_Ka_v5.fits',
            'path_noise_simulations': path_noise_wmap + '33/',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_33ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_33ghz_0001.fits',
        },
        '41': {
            'freq': 40.77 * u.GHz,
            'fwhm': 30.6 * u.arcmin,
            'noise_I': 2.188 * u.mK,
            'noise_QU': 2.197 * u.mK,
            'path': path_map_wmap + 'wmap_band_iqumap_r9_9yr_Q_v5.fits',
            'path_simulated': path_sim_wmap + 'map_WMAP_41_nside512_beamconv.fits',
            'beam': path_beam_wmap + 'wmap_ampl_bl_Q_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + 'wmap_iqumap_r9_hmdm_Q_v5.fits',
            'half_1': path_half1_wmap + 'wmap_iqumap_r9_yr1to4_Q_v5.fits',
            'half_2': path_half2_wmap + 'wmap_iqumap_r9_yr5to9_Q_v5.fits',
            'path_noise_simulations': path_noise_wmap + '41/',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_41ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_41ghz_0001.fits',
        },
        '61': {
            'freq': 60.12 * u.GHz,
            'fwhm': 21.0 * u.arcmin,
            'noise_I': 3.131 * u.mK,
            'noise_QU': 3.141 * u.mK,
            'path': path_map_wmap + 'wmap_band_iqumap_r9_9yr_V_v5.fits',
            'path_simulated': path_sim_wmap + 'map_WMAP_61_nside512_beamconv.fits',
            'beam': path_beam_wmap + 'wmap_ampl_bl_V_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + 'wmap_iqumap_r9_hmdm_V_v5.fits',
            'half_1': path_half1_wmap + 'wmap_iqumap_r9_yr1to4_V_v5.fits',
            'half_2': path_half2_wmap + 'wmap_iqumap_r9_yr5to9_V_v5.fits',
            'path_noise_simulations': path_noise_wmap + '61/',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_61ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_61ghz_0001.fits',
        },
        '94': {
            'freq': 92.87 * u.GHz,
            'fwhm': 13.2 * u.arcmin,
            'noise_I': 6.544 * u.mK,
            'noise_QU': 6.560 * u.mK,
            'path': path_map_wmap + 'wmap_band_iqumap_r9_9yr_W_v5.fits',
            'path_simulated': path_sim_wmap + 'map_WMAP_94_nside512_beamconv.fits',
            'beam': path_beam_wmap + 'wmap_ampl_bl_W_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + 'wmap_iqumap_r9_hmdm_W_v5.fits',
            'half_1': path_half1_wmap + 'wmap_iqumap_r9_yr1to4_W_v5.fits',
            'half_2': path_half2_wmap + 'wmap_iqumap_r9_yr5to9_W_v5.fits',
            'path_noise_simulations': path_noise_wmap + '94/',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_94ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_94ghz_0001.fits',
        },

        '23_1': {
            'freq': 22.72 * u.GHz,
            'fwhm': 52.8 * u.arcmin,
            'noise_I': 1.437 * u.mK,      
            'noise_QU': 1.456 * u.mK,
            'path': path_map_wmap + 'wmap_iqumap_r9_9yr_K1_v5.fits',
            'path_simulated': path_sim_wmap + '',
            'beam': path_beam_wmap + 'wmap_ampl_bl_K1_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + '',
            'half_1': path_half1_wmap + 'wmap_iqumap_r9_yr1to4_K_v5.fits',
            'half_2': path_half2_wmap + 'wmap_iqumap_r9_yr5to9_K_v5.fits',
            'path_noise_simulations': path_noise_wmap + '',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_23_1ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_23_1ghz_0001.fits',
        },
        '33_1': {
            'freq': 32.98 * u.GHz,
            'fwhm': 39.6 * u.arcmin,
            'noise_I': 1.470 * u.mK,
            'noise_QU': 1.490 * u.mK,
            'path': path_map_wmap + 'wmap_iqumap_r9_9yr_Ka1_v5.fits',
            'path_simulated': path_sim_wmap + '',
            'beam': path_beam_wmap + 'wmap_ampl_bl_Ka1_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + '',
            'half_1': path_half1_wmap + '',
            'half_2': path_half2_wmap + '',
            'path_noise_simulations': path_noise_wmap + '',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_33_1ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_33_1ghz_0001.fits',
        },
        '41_1': {
            'freq': 40.77 * u.GHz,
            'fwhm': 30.6 * u.arcmin,
            'noise_I': 2.254 * u.mK,
            'noise_QU': 2.280 * u.mK,
            'path': path_map_wmap + 'wmap_iqumap_r9_9yr_Q1_v5.fits',
            'path_simulated': path_sim_wmap + '',
            'beam': path_beam_wmap + 'wmap_ampl_bl_Q1_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + '',
            'half_1': path_half1_wmap + '',
            'half_2': path_half2_wmap + '',
            'path_noise_simulations': path_noise_wmap + '',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_41_1ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_41_1ghz_0001.fits',
        },
        '41_2': {
            'freq': 40.56 * u.GHz,
            'fwhm': 30.6 * u.arcmin,
            'noise_I': 2.140 * u.mK,
            'noise_QU': 2.164 * u.mK,
            'path': path_map_wmap + 'wmap_iqumap_r9_9yr_Q2_v5.fits',
            'path_simulated': path_sim_wmap + '',
            'beam': path_beam_wmap + 'wmap_ampl_bl_Q2_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + '',
            'half_1': path_half1_wmap + '',
            'half_2': path_half2_wmap + '',
            'path_noise_simulations': path_noise_wmap + '',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_41_2ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_41_2ghz_0001.fits',
        },
        '61_1': {
            'freq': 60.12 * u.GHz,
            'fwhm': 21.0 * u.arcmin,
            'noise_I': 3.319 * u.mK,
            'noise_QU': 3.348 * u.mK,
            'path': path_map_wmap + 'wmap_iqumap_r9_9yr_V1_v5.fits',
            'path_simulated': path_sim_wmap + '',
            'beam': path_beam_wmap + 'wmap_ampl_bl_V1_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + '',
            'half_1': path_half1_wmap + '',
            'half_2': path_half2_wmap + '',
            'path_noise_simulations': path_noise_wmap + '',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_61_1ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_61_1ghz_0001.fits',
        },
        '61_2': {
            'freq': 60.00 * u.GHz,
            'fwhm': 21.0 * u.arcmin,
            'noise_I': 2.955 * u.mK,
            'noise_QU': 2.979 * u.mK,
            'path': path_map_wmap + 'wmap_iqumap_r9_9yr_V2_v5.fits',
            'path_simulated': path_sim_wmap + '',
            'beam': path_beam_wmap + 'wmap_ampl_bl_V2_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + '',
            'half_1': path_half1_wmap + '',
            'half_2': path_half2_wmap + '',
            'path_noise_simulations': path_noise_wmap + '',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_61_2ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_61_2ghz_0001.fits',
        },
        '94_1': {
            'freq': 92.87 * u.GHz,
            'fwhm': 13.2 * u.arcmin,
            'noise_I': 5.906 * u.mK,
            'noise_QU': 5.940 * u.mK,
            'path': path_map_wmap + 'wmap_iqumap_r9_9yr_W1_v5.fits',
            'path_simulated': path_sim_wmap + '',
            'beam': path_beam_wmap + 'wmap_ampl_bl_W1_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + '',
            'half_1': path_half1_wmap + '',
            'half_2': path_half2_wmap + '',
            'path_noise_simulations': path_noise_wmap + '',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_94_1ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_94_1ghz_0001.fits',
        },
        '94_2': {
            'freq': 93.43 * u.GHz,
            'fwhm': 13.2 * u.arcmin,
            'noise_I': 6.572 * u.mK,
            'noise_QU': 6.612 * u.mK,
            'path': path_map_wmap + 'wmap_iqumap_r9_9yr_W2_v5.fits',
            'path_simulated': path_sim_wmap + '',
            'beam': path_beam_wmap + 'wmap_ampl_bl_W2_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + '',
            'half_1': path_half1_wmap + '',
            'half_2': path_half2_wmap + '',
            'path_noise_simulations': path_noise_wmap + '',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_94_2ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_94_2ghz_0001.fits',
        },
        '94_3': {
            'freq': 92.44 * u.GHz,
            'fwhm': 13.2 * u.arcmin,
            'noise_I': 6.941 * u.mK,
            'noise_QU': 6.983 * u.mK,
            'path': path_map_wmap + 'wmap_iqumap_r9_9yr_W3_v5.fits',
            'path_simulated': path_sim_wmap + '',
            'beam': path_beam_wmap + 'wmap_ampl_bl_W3_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + '',
            'half_1': path_half1_wmap + '',
            'half_2': path_half2_wmap + '',
            'path_noise_simulations': path_noise_wmap + '',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_94_3ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_94_3ghz_0001.fits',
        },
        '94_4': {
            'freq': 93.22 * u.GHz,
            'fwhm': 13.2 * u.arcmin,
            'noise_I': 6.778 * u.mK,
            'noise_QU': 6.840 * u.mK,
            'path': path_map_wmap + 'wmap_iqumap_r9_9yr_W4_v5.fits',
            'path_simulated': path_sim_wmap + '',
            'beam': path_beam_wmap + 'wmap_ampl_bl_W4_9yr_v5p1.txt',
            'hmdm': path_hmdm_wmap + '',
            'half_1': path_half1_wmap + '',
            'half_2': path_half2_wmap + '',
            'path_noise_simulations': path_noise_wmap + '',
            'path_white_noise_simulations': path_white_noise_wmap + '',
            'noise_simulation_1': 'white_noise_94_4ghz_0001.fits',
            'white_noise_simulation_1': 'white_noise_94_4ghz_0001.fits',
        },
    },
    'Planck': {
        '30': {
            'freq': 28.4 * u.GHz,
            'fwhm': 32.29 * u.arcmin,
            'noise_I': u.K,
            'noise_QU': u.K,
            'path': path_map_planck + 'LFI_SkyMap_030-BPassCorrected_1024_R3.00_full.fits',
            'path_simulated': path_sim_planck + 'map_Planck_30_nside512_beamconv.fits',
            'beam': path_beam_planck + 'LFI_RIMO_R3.31.fits',
            'hmdm': path_hmdm_planck + 'LFI_SkyMap_030-BPassCorrected_1024_R3.00_hmdm.fits',
            'half_1': path_half1_planck + 'LFI_SkyMap_030-BPassCorrected_1024_R3.00_full-ringhalf-1.fits',
            'half_2': path_half2_planck + 'LFI_SkyMap_030-BPassCorrected_1024_R3.00_full-ringhalf-2.fits',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': 'ffp10_noise_030_full_map_mc_00001.fits',
            'white_noise_simulation_1': 'white_noise_30ghz_0001.fits',
        },
        '44': {
            'freq': 44.1 * u.GHz,
            'fwhm': 26.99 * u.arcmin,
            'noise_I': u.K,
            'noise_QU': u.K,
            'path': path_map_planck + 'LFI_SkyMap_044-BPassCorrected_1024_R3.00_full.fits',
            'path_simulated': path_sim_planck + 'map_Planck_44_nside512_beamconv.fits',
            'beam': path_beam_planck + 'LFI_RIMO_R3.31.fits',
            'hmdm': path_hmdm_planck + 'LFI_SkyMap_044-BPassCorrected_1024_R3.00_hmdm.fits',
            'half_1': path_half1_planck + 'LFI_SkyMap_044-BPassCorrected_1024_R3.00_full-ringhalf-1.fits',
            'half_2': path_half2_planck + 'LFI_SkyMap_044-BPassCorrected_1024_R3.00_full-ringhalf-2.fits',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': 'ffp10_noise_044_full_map_mc_00001.fits',
            'white_noise_simulation_1': 'white_noise_44ghz_0001.fits',
        },
        '70': {
            'freq': 70.4 * u.GHz,
            'fwhm': 13.22 * u.arcmin,
            'noise_I': u.K,
            'noise_QU': u.K,
            'path': path_map_planck + 'LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits',
            'path_simulated': path_sim_planck + 'map_Planck_70_nside512_beamconv.fits',
            'beam': path_beam_planck + 'LFI_RIMO_R3.31.fits',
            'hmdm': path_hmdm_planck + 'LFI_SkyMap_070-BPassCorrected_1024_R3.00_hmdm.fits',
            'half_1': path_half1_planck + 'LFI_SkyMap_070-BPassCorrected_1024_R3.00_full-ringhalf-1.fits',
            'half_2': path_half2_planck + 'LFI_SkyMap_070-BPassCorrected_1024_R3.00_full-ringhalf-2.fits',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': 'ffp10_noise_070_full_map_mc_00001.fits',
            'white_noise_simulation_1': 'white_noise_70ghz_0001.fits',
        },
        '100': {
            'freq': 101.31 * u.GHz,
            'fwhm': 9.680 * u.arcmin,
            'noise_I': u.K,
            'noise_QU': u.K,  
            'path': path_map_planck + 'HFI_SkyMap_100_2048_R3.01_full.fits',
            'path_simulated': path_sim_planck + 'map_Planck_100_nside512_beamconv.fits',
            'beam': path_beam_planck + 'BeamWf_HFI_R3.01/Bl_TEB_R3.01_fullsky_100x100.fits',
            'hmdm': path_hmdm_planck + 'HFI_SkyMap_100-BPassCorrected_2048_R3.01_hmdm.fits',
            'half_1': path_half1_planck + 'HFI_SkyMap_100_2048_R3.01_halfmission-1.fits',
            'half_2': path_half2_planck + 'HFI_SkyMap_100_2048_R3.01_halfmission-2.fits',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': 'ffp10_noise_100_full_map_mc_00001.fits',
            'white_noise_simulation_1': 'white_noise_100ghz_0001.fits',
        },
        '143': {
            'freq': 142.709 * u.GHz,
            'fwhm': 7.300 * u.arcmin,
            'noise_I': u.K,
            'noise_QU': u.K,
            'path': path_map_planck + 'HFI_SkyMap_143_2048_R3.01_full.fits',
            'path_simulated': path_sim_planck + 'map_Planck_143_nside512_beamconv.fits',
            'beam': path_beam_planck + 'BeamWf_HFI_R3.01/Bl_TEB_R3.01_fullsky_143x143.fits',
            'hmdm': path_hmdm_planck + 'HFI_SkyMap_143-BPassCorrected_2048_R3.01_hmdm.fits',
            'half_1': path_half1_planck + 'HFI_SkyMap_143_2048_R3.01_halfmission-1.fits',
            'half_2': path_half2_planck + 'HFI_SkyMap_143_2048_R3.01_halfmission-2.fits',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': 'ffp10_noise_143_full_map_mc_00001.fits',
            'white_noise_simulation_1': 'white_noise_143ghz_0001.fits',
        },
        '217': {
            'freq': 221.914 * u.GHz,
            'fwhm': 5.020 * u.arcmin,
            'noise_I': u.K,
            'noise_QU': u.K,
            'path': path_map_planck + 'HFI_SkyMap_217_2048_R3.01_full.fits',
            'path_simulated': path_sim_planck + 'map_Planck_217_nside512_beamconv.fits',
            'beam': path_beam_planck + 'BeamWf_HFI_R3.01/Bl_TEB_R3.01_fullsky_217x217.fits',
            'hmdm': path_hmdm_planck + 'HFI_SkyMap_217-BPassCorrected_2048_R3.01_hmdm.fits',
            'half_1': path_half1_planck + 'HFI_SkyMap_217_2048_R3.01_halfmission-1.fits',
            'half_2': path_half2_planck + 'HFI_SkyMap_217_2048_R3.01_halfmission-2.fits',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': 'ffp10_noise_217_full_map_mc_00001.fits',
            'white_noise_simulation_1': 'white_noise_217ghz_0001.fits',
        },
        '353': {
            'freq': 361.289 * u.GHz,
            'fwhm': 4.940 * u.arcmin,
            'noise_I': u.K,
            'noise_QU': u.K,
            'path': path_map_planck + 'HFI_SkyMap_353-psb_2048_R3.01_full.fits',
            'path_simulated': path_sim_planck + 'map_Planck_353_nside512_beamconv.fits',
            'beam': path_beam_planck + 'BeamWf_HFI_R3.01/Bl_TEB_R3.01_fullsky_353x353.fits',
            'hmdm': path_hmdm_planck + 'HFI_SkyMap_353-BPassCorrected_2048_R3.01_hmdm.fits',
            'half_1': path_half1_planck + 'HFI_SkyMap_353-psb_2048_R3.01_halfmission-1.fits',
            'half_2': path_half2_planck + 'HFI_SkyMap_353-psb_2048_R3.01_halfmission-2.fits',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': 'ffp10_noise_353_psb_full_map_mc_00001.fits',
            'white_noise_simulation_1': 'white_noise_353ghz_0001.fits',
        },

        '30_pr4': {
            'freq': 28.4 * u.GHz,
            'fwhm': 32.29 * u.arcmin,
            'noise_I': u.K,
            'noise_QU': u.K,
            'path': path_map_planck + 'LFI_SkyMap_030-BPassCorrected_1024_R4.00_full.fits',
            'path_simulated': path_sim_planck + 'Planck_30_pr4GHz_n512_fwhm_3229.fits',
            'beam': path_beam_planck + '',
            'hmdm': path_hmdm_planck + '',
            'half_1': path_half1_planck + '',
            'half_2': path_half2_planck + '',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': 'npipe6v20_noise_030_mc_00200.fits',
            'white_noise_simulation_1': 'white_noise_30_pr4ghz_0001.fits',
        },
        '44_pr4': {
            'freq': 44.1 * u.GHz,
            'fwhm': 26.99 * u.arcmin,
            'noise_I': u.K,
            'noise_QU': u.K,
            'path': path_map_planck + 'LFI_SkyMap_044-BPassCorrected_1024_R4.00_full.fits',
            'path_simulated': path_sim_planck + 'Planck_44_pr4GHz_n512_fwhm_2699.fits',
            'beam': path_beam_planck + '',
            'hmdm': path_hmdm_planck + '',
            'half_1': path_half1_planck + '',
            'half_2': path_half2_planck + '',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': '',
            'white_noise_simulation_1': 'white_noise_44_pr4ghz_0001.fits',
        },
        '70_pr4': {
            'freq': 70.4 * u.GHz,
            'fwhm': 13.22 * u.arcmin,
            'noise_I': u.K,
            'noise_QU': u.K,
            'path': path_map_planck + 'LFI_SkyMap_070-BPassCorrected_1024_R4.00_full.fits',
            'path_simulated': path_sim_planck + 'Planck_70_pr4GHz_n512_fwhm_1322.fits',
            'beam': path_beam_planck + '',
            'hmdm': path_hmdm_planck + '',
            'half_1': path_half1_planck + '',
            'half_2': path_half2_planck + '',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': '',
            'white_noise_simulation_1': 'white_noise_70_pr4ghz_0001.fits',
        },
        '100_pr4': {
            'freq': 101.31 * u.GHz,
            'fwhm': 9.680 * u.arcmin,
            'noise_I': u.K,
            'noise_QU': u.K,  
            'path': path_map_planck + 'HFI_SkyMap_100-BPassCorrected_2048_R4.00_full.fits',
            'path_simulated': path_sim_planck + 'Planck_100_pr4GHz_n512_fwhm_0968.fits',
            'beam': path_beam_planck + '',
            'hmdm': path_hmdm_planck + '',
            'half_1': path_half1_planck + '',
            'half_2': path_half2_planck + '',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': '',
            'white_noise_simulation_1': 'white_noise_100_pr4ghz_0001.fits',
        },
        '143_pr4': {
            'path': path_map_planck + 'HFI_SkyMap_143_2048_R3.01_full.fits',
            'path_simulated': path_sim_planck + 'PYSM/map_Planck_143_nside512_beamconv.fits',
            'beam': path_beam_planck + 'Bl_TEB_R3.01_fullsky_143x143.fits',
            'hmdm': path_hmdm_planck + 'HFI_SkyMap_143-BPassCorrected_2048_R3.01_hmdm.fits',
            'half_1': path_half1_planck + 'HFI_SkyMap_143_2048_R3.01_halfmission-1.fits',
            'half_2': path_half2_planck + 'HFI_SkyMap_143_2048_R3.01_halfmission-2.fits',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'half_1': path_half1_planck + '',
            'half_2': path_half2_planck + '',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': '',
            'white_noise_simulation_1': 'white_noise_143_pr4ghz_0001.fits',
        },
        '217_pr4': {
            'path': path_map_planck + 'HFI_SkyMap_217_2048_R3.01_full.fits',
            'path_simulated': path_sim_planck + 'PYSM/map_Planck_217_nside512_beamconv.fits',
            'beam': path_beam_planck + 'Bl_TEB_R3.01_fullsky_217x217.fits',
            'hmdm': path_hmdm_planck + 'HFI_SkyMap_217-BPassCorrected_2048_R3.01_hmdm.fits',
            'half_1': path_half1_planck + 'HFI_SkyMap_217_2048_R3.01_halfmission-1.fits',
            'half_2': path_half2_planck + 'HFI_SkyMap_217_2048_R3.01_halfmission-2.fits',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'half_1': path_half1_planck + '',
            'half_2': path_half2_planck + '',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': '',
            'white_noise_simulation_1': 'white_noise_217_pr4ghz_0001.fits',
        },
        '353_pr4': {
            'path': path_map_planck + 'HFI_SkyMap_353-psb_2048_R3.01_full.fits',
            'path_simulated': path_sim_planck + 'PYSM/map_Planck_353_nside512_beamconv.fits',
            'beam': path_beam_planck + 'Bl_TEB_R3.01_fullsky_353x353.fits',
            'hmdm': path_hmdm_planck + 'HFI_SkyMap_353-BPassCorrected_2048_R3.01_hmdm.fits',
            'half_1': path_half1_planck + 'HFI_SkyMap_353-psb_2048_R3.01_halfmission-1.fits',
            'half_2': path_half2_planck + 'HFI_SkyMap_353-psb_2048_R3.01_halfmission-2.fits',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'half_1': path_half1_planck + '',
            'half_2': path_half2_planck + '',
            'path_noise_simulations': path_noise_planck + '',
            'path_white_noise_simulations': path_white_noise_planck + '',
            'noise_simulation_1': '',
            'white_noise_simulation_1': 'white_noise_353_pr4ghz_0001.fits',
        },
    }
}


masks = {
    'QUIJOTE': {
        'lowdec_satband': {
            'name': 'quijote',
            'fsky': 0.,
            'path': path_masks + 'quijote/mask_quijote_ncp_lowdec_satband_nside512.fits'
        },
        'satband': {
            'name': 'quijote_satband',
            'fsky': 0.,
            'path': path_masks + 'quijote/mask_quijote_satband_ncp86_nside512.fits'
        },
    },

    'QUIJOTE_galcut': {
        'galcut5': {
            'name': 'quijote_galcut5',
            'fsky': 0.3319,
            'path': path_masks + 'quijote_galcut/mask_lowdec_satband_galcut5_0mk_apodC2_5.fits'
        },
        'galcut10': {
            'name': 'quijote_galcut10',
            'fsky': 0.2968,
            'path': path_masks + 'quijote_galcut/mask_lowdec_satband_galcut10_0mk_apodC2_5.fits'
        },
        'galcut15': {
            'name': 'quijote_galcut15',
            'fsky': 0.2643,
            'path': path_masks + 'quijote_galcut/mask_lowdec_satband_galcut15_0mk_apodC2_5.fits'
        },
        'galcut20': {
            'name': 'quijote_galcut20',
            'fsky': 0.2332,
            'path': path_masks + 'quijote_galcut/mask_lowdec_satband_galcut20_0mk_apodC2_5.fits'
        },
        'galcut10_10mk': {
            'name': 'quijote_galcut10_10mk',
            'fsky': 0.2940,
            'path': path_masks + 'quijote_galcut/mask_lowdec_satband_galcut10_10mk_apodC2_5.fits'
        },
    },

    'north_south': {
        'north': {
            'name': 'quijote_north',
            'fsky': 0.,
            'path': path_masks + 'north_south/mask_north_dec_6_70_apodC2_5_pc.fits'
        },
        'south': {
            'name': 'quijote_south',
            'fsky': 0.,
            'path': path_masks + 'north_south/mask_south_dec_6_70_apodC2_5_pc.fits'
        },
    },

    'specific_regions': {
        'fan': {
            'name': 'fan',
            'fsky': 0.0315,
            'path': path_masks + 'specific_regions/mask_lowdec_satband_galcut0_0mk_fan_apodC2_5.fits'
        },
        'cygnus': {
            'name': 'cygnus',
            'fsky': 0.0272,
            'path': path_masks + 'specific_regions/mask_lowdec_satband_galcut0_0mk_cygnus_apodC2_5.fits'
        },
        'nps': {
            'name': 'nps',
            'fsky': 0.0250,
            'path': path_masks + 'specific_regions/mask_lowdec_satband_galcut0_0mk_nps_apodC2_5.fits'
        },
        'north': {
            'name': 'north',
            'fsky': 0.1730,
            'path': path_masks + 'specific_regions/mask_lowdec_satband_galcut0_0mk_north_apodC2_5.fits'
        },
        'south': {
            'name': 'south',
            'fsky': 0.0876,
            'path': path_masks + 'specific_regions/mask_lowdec_satband_galcut0_0mk_south_apodC2_5.fits'
        },
    },
    
}

color_corrections = {
    'Planck_HFI_dust_bps': path_cc + 'c_td_10-40_beta_hfi_bps_pr3.fits',
    'Planck_HFI_dust': path_cc + 'c_td_10-40_beta_hfi_pr3.fits',
    'cc_polynoms': path_cc + 'cc_polynoms.fits'

}