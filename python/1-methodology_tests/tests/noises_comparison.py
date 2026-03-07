#%%

import os
import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

import functions
from data import data, masks

# ============================================================
# Configuration (mirror of run_all_code.ipynb)
# ============================================================
nside       = 512
n_sim       = 100
name_suffix = '_full_bin_20-199'
out_path    = '/home/pablo/Desktop/master/tfm/spectra/'

mask_select = masks['QUIJOTE_galcut']['galcut10']
mask_name   = mask_select['name']

quijote_bands = ['11', '13', '17', '19']
wmap_bands    = ['23', '33', '41', '61', '94']
planck_bands  = ['30', '44', '70', '100', '143', '217', '353']
band_list     = quijote_bands + wmap_bands + planck_bands

# Bands shown in the figure (subset for the study)
plot_bands = ['11', '23', '30', '100', '353']

path_avg_std_skyplusnoise = os.path.join(
    out_path,
    f'spectra_avg_std_{mask_name}_avg_std{n_sim}_skyplusnoise{name_suffix}.fits'
)
path_avg_std_noise = os.path.join(
    out_path,
    f'spectra_avg_std_{mask_name}_avg_std{n_sim}_noise{name_suffix}.fits'
)

save_fig_path = (
    f'/home/pablo/Desktop/master/tfm/figures/spectra/'
    f'noise_comparison_{mask_name}{name_suffix}.pdf'
)
 
# also save the printed table to a txt file
save_txt_path = (
    f'/home/pablo/Desktop/master/tfm/figures/spectra/'
    f'noise_comparison_{mask_name}{name_suffix}.txt'
)
os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
fout = open(save_txt_path, 'w')

# ============================================================
# Load data
# ============================================================
print('Loading sky+noise simulations ...')
avg_sn = functions.read_spectra_from_fits(path_avg_std_skyplusnoise, band_list)

print('Loading noise-only simulations ...')
avg_n  = functions.read_spectra_from_fits(path_avg_std_noise,        band_list)

# Auto-spectrum keys only
auto_keys = [f'{b}_{b}' for b in band_list]
ell_eff   = np.array(avg_sn[auto_keys[0]]['ell_eff']['MEAN'])
n_bins    = len(ell_eff)

modes = ['EE', 'BB']

# ============================================================
# Printed table + collect arrays for plotting
# ============================================================
# store: data_store[mode][band] = dict with arrays
data_store = {m: {} for m in modes}

for mode in modes:
    sep_line = '\n' + '=' * 82
    print(sep_line)
    print(f'  MODE: {mode}')
    print('=' * 82)
    fout.write(sep_line + '\n')
    fout.write(f'  MODE: {mode}\n')
    fout.write('=' * 82 + '\n')

    col_w = 8
    hdr = (f'{"Band":<{col_w}} {"bin":>4}  {"ell_eff":>8}  '
           f'{"sigma_cl":>14}  {"sigma_nl":>14}  '
           f'{"sigma_tot":>14}  {"ratio":>8}')
    print(hdr)
    print('-' * len(hdr))
    fout.write(hdr + '\n')
    fout.write('-' * len(hdr) + '\n')

    all_ratios = []

    for key in auto_keys:
        band  = key.split('_')[0]
        ell   = np.array(avg_sn[key]['ell_eff']['MEAN'])
        sn    = np.array(avg_sn[key][mode]['STD'])
        n_    = np.array(avg_n [key][mode]['STD'])
        tot   = np.sqrt(sn**2 + n_**2)
        ratio = np.where(tot > 0, sn / tot, np.nan)

        data_store[mode][band] = dict(ell=ell, sn=sn, n_=n_, tot=tot, ratio=ratio)
        all_ratios.append(ratio)

        if band == '11':
            for i in range(len(ell)):
                line = (f'{band:<{col_w}} {i+1:>4}  {ell[i]:>8.1f}  '
                        f'{sn[i]:>14.4e}  {n_[i]:>14.4e}  '
                        f'{tot[i]:>14.4e}  {ratio[i]:>8.4f}')
                print(line)
                fout.write(line + '\n')

    mean_ratio = np.nanmean(np.concatenate(all_ratios))
    msg = f'\n  Mean ratio sigma_cl / sigma_tot  [{mode}] = {mean_ratio:.4f}'
    print(msg)
    fout.write(msg + '\n')

# ============================================================
# Figure  — 5 bands × 2 modes (EE top row, BB bottom row)
# ============================================================
ncols = len(plot_bands)   # 5
nrows = 2                 # EE / BB

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4.2 * ncols, 4.0 * nrows),
    sharex=False,
    constrained_layout=True
)

mode_labels = {'EE': 'EE', 'BB': 'BB'}

for row, mode in enumerate(modes):
    for col, band in enumerate(plot_bands):
        ax = axes[row, col]
        d  = data_store[mode][band]

        ax.plot(d['ell'], d['sn'],  'o-',  color='steelblue', ms=5, lw=1.6,
                label=r'$\sigma_{\rm C_{\ell}}$')
        ax.plot(d['ell'], d['n_'], 's--', color='k',  ms=5, lw=1.6,
                label=r'$\sigma_{\rm N_{\ell}}$')

        ax.set_yscale('log')
        ax.set_title(f'{band} GHz — {mode_labels[mode]}')
        ax.set_xlabel(r'$\ell_{\rm eff}$')
        if col == 0:
            ax.set_ylabel(r'$\sigma\;[\mathrm{mK}^2_{\rm RJ}]$')
        ax.tick_params()
        ax.legend(frameon=False)


os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
fig.savefig(save_fig_path, bbox_inches='tight')
msg_fig = f'\nFigure saved → {save_fig_path}'
print(msg_fig)
fout.write(msg_fig + '\n')
fout.close()
plt.show()
