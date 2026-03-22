#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines

# ---------------------------------------------------------------
# Global plotting style
# ---------------------------------------------------------------
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.size': 12,
})

# ---------------------------------------------------------------
# A_s data extracted from the three bin-to-bin tables (QJ+WP+Pl)
# Units: 10^{-3} muK^2
# Structure: DATA[mask_key][mode] = dict(ell, val, err_up, err_dn)
# ---------------------------------------------------------------
DATA = {
    'galcut10': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([294.505, 37.590, 13.055, 6.238, 3.519, 2.535, 1.980, 1.299, 1.070]),
            err_up = np.array([3.295, 1.238, 0.753, 0.606, 0.507, 0.477, 0.583, 0.524, 0.658]),
            err_dn = np.array([3.397, 1.322, 0.759, 0.639, 0.523, 0.500, 0.593, 0.502, 0.617]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([47.841, 9.572, 1.185, 0.718, 0.829, 1.101, 0.152, 1.057, 1.095]),
            err_up = np.array([1.913, 0.753, 0.492, 0.438, 0.454, 0.417, 0.351, 0.486, 0.653]),
            err_dn = np.array([1.914, 0.772, 0.516, 0.408, 0.446, 0.440, 0.129, 0.483, 0.586]),
        ),
    },
    'galcut15': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([276.355, 32.205, 10.944, 5.791, 3.234, 2.109, 1.952, 1.552, 0.815]),
            err_up = np.array([3.451, 1.230, 0.757, 0.602, 0.489, 0.515, 0.606, 0.564, 0.689]),
            err_dn = np.array([3.415, 1.292, 0.774, 0.638, 0.500, 0.508, 0.609, 0.546, 0.580]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([41.336, 8.650, 0.866, 0.832, 0.953, 1.012, 0.186, 1.104, 1.859]),
            err_up = np.array([2.018, 0.794, 0.520, 0.487, 0.501, 0.441, 0.382, 0.471, 0.642]),
            err_dn = np.array([2.036, 0.812, 0.506, 0.471, 0.506, 0.437, 0.156, 0.454, 0.660]),
        ),
    },
    'galcut20': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([263.649, 32.168, 10.382, 5.343, 2.461, 1.955, 2.391, 1.769, 0.829]),
            err_up = np.array([3.842, 1.267, 0.738, 0.607, 0.513, 0.504, 0.584, 0.586, 0.681]),
            err_dn = np.array([3.803, 1.293, 0.721, 0.608, 0.513, 0.498, 0.601, 0.569, 0.572]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([39.584, 8.077, 0.888, 0.747, 1.143, 0.993, 0.235, 1.332, 2.067]),
            err_up = np.array([1.976, 0.815, 0.547, 0.520, 0.550, 0.471, 0.461, 0.527, 0.669]),
            err_dn = np.array([1.985, 0.851, 0.533, 0.469, 0.577, 0.454, 0.198, 0.510, 0.686]),
        ),
    },
}

# ---------------------------------------------------------------
# Visual encoding
# ---------------------------------------------------------------
MASK_STYLES = {
    'galcut10': dict(color='steelblue',  marker='o', label=r'$\pm10\degree$', offset=-1.7),
    'galcut15': dict(color='w',          marker='^', label=r'$\pm15\degree$', offset= 0),
    'galcut20': dict(color='goldenrod',  marker='s', label=r'$\pm20\degree$', offset=+1.7),
}

# ---------------------------------------------------------------
# Figure: 2 stacked panels (EE top, BB bottom), shared x-axis
# ---------------------------------------------------------------
fig, (ax_EE, ax_BB) = plt.subplots(
    2, 1, figsize=(15, 6), sharex=True,
    gridspec_kw={'hspace': 0.15})

ell_ticks = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float)

legend_handles = [
    mlines.Line2D([], [],
                  color=mstyle['color'],
                  marker=mstyle['marker'],
                  ls='none', ms=7,
                  label=mstyle['label'])
    for mstyle in MASK_STYLES.values()
]

for ax, mode, panel_label in [
        (ax_EE, 'EE', 'EE'),
        (ax_BB, 'BB', 'BB')]:

    # ax.axhline(0, color='grey', lw=0.8, ls='--', zorder=1)

    for mask_key, mstyle in MASK_STYLES.items():
        d    = DATA[mask_key][mode]
        xpos = d['ell'] + mstyle['offset']
        ax.errorbar(
            xpos,
            d['val'],
            yerr=[d['err_dn'], d['err_up']],
            fmt=mstyle['marker'],
            color=mstyle['color'],
            ms=6,
            capsize=3,
            capthick=1.2,
            lw=1.2,
            zorder=3,
        )

    if mode == 'EE':
        ax.set_ylabel(r'$A^{\rm EE}_{\rm s}\ [10^{-3}\,\mu{\rm K}^2]$', fontsize=15)
    else:
        ax.set_ylabel(r'$A^{\rm BB}_{\rm s}\ [10^{-3}\,\mu{\rm K}^2]$', fontsize=15)
    ax.set_xlim(10, 220)

    ax.set_yscale('log')

# legend inside the EE panel (top right)
ax_EE.legend(
    handles=legend_handles,
    loc='upper right',
    frameon=False,
    framealpha=0.85,
    edgecolor='grey',
    fontsize=15,
    handletextpad=0.4,
    borderpad=0.6,
)

ax_BB.set_xticks(ell_ticks)
ax_BB.set_xticklabels([str(int(v)) for v in ell_ticks])
ax_BB.set_xlabel(r'$\ell$', fontsize=15)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Save PDF
# ---------------------------------------------------------------
out_dir = '/home/pablo/Desktop/master/tfm/figures/spectra/'
os.makedirs(out_dir, exist_ok=True)
# fig.savefig(os.path.join(out_dir, 'As_vs_ell_bin_to_bin.pdf'), bbox_inches='tight')
print('Saved As_vs_ell_bin_to_bin.pdf')

# ---------------------------------------------------------------
# Transparent PNG helper (black -> white)
# ---------------------------------------------------------------
def save_transparent_white(fig, path, dpi=300):
    def is_black(c):
        try:
            if isinstance(c, str):
                return c.lower() in ('k', 'black', '#000000')
            if hasattr(c, '__len__'):
                rgba = mpl.colors.to_rgba(c)
                return rgba[0] < 1e-6 and rgba[1] < 1e-6 and rgba[2] < 1e-6
        except Exception:
            pass
        return False

    fig.patch.set_alpha(0.0)
    for a in fig.axes:
        try:
            a.set_facecolor('none')
        except Exception:
            pass
        for attr in (a.title, a.xaxis.label, a.yaxis.label):
            try:
                if is_black(attr.get_color()):
                    attr.set_color('white')
            except Exception:
                pass
        try:
            a.tick_params(colors='white')
            for lbl in a.get_xticklabels() + a.get_yticklabels():
                if is_black(lbl.get_color()):
                    lbl.set_color('white')
        except Exception:
            pass
        for spine in a.spines.values():
            try:
                if is_black(spine.get_edgecolor()):
                    spine.set_edgecolor('white')
            except Exception:
                pass
        leg = a.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                try:
                    if is_black(txt.get_color()):
                        txt.set_color('white')
                except Exception:
                    pass
            try:
                frame = leg.get_frame()
                frame.set_alpha(0.0)
                if is_black(frame.get_edgecolor()):
                    frame.set_edgecolor('white')
            except Exception:
                pass
        for line in a.get_lines():
            try:
                if is_black(line.get_color()):
                    line.set_color('white')
            except Exception:
                pass
    for txt in fig.findobj(mpl.text.Text):
        try:
            if is_black(txt.get_color()):
                txt.set_color('white')
        except Exception:
            pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, transparent=True, facecolor='none',
                bbox_inches='tight', dpi=dpi)
    print('Saved transparent PNG to', path)

ppt_dir = '/home/pablo/Desktop/master/tfm/figures_ppt/spectra/'
# save_transparent_white(
#     fig,
#     os.path.join(ppt_dir, 'As_vs_ell_bin_to_bin_transparent.png'),
#     dpi=300)

plt.show()

# ---------------------------------------------------------------
# A_d data extracted from the three bin-to-bin tables (QJ+WP+Pl)
# Units: 10^{-3} muK^2
# ---------------------------------------------------------------
DATA_DUST = {
    'galcut10': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([49.086, 7.887, 4.798, 2.673, 1.555, 1.017, 0.679, 0.479, 0.404]),
            err_up = np.array([0.270, 0.077, 0.040, 0.022, 0.016, 0.013, 0.010, 0.008, 0.007]),
            err_dn = np.array([0.261, 0.084, 0.041, 0.023, 0.017, 0.013, 0.010, 0.008, 0.007]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([22.650, 6.111, 3.378, 2.010, 1.325, 0.826, 0.527, 0.421, 0.365]),
            err_up = np.array([0.173, 0.052, 0.023, 0.016, 0.011, 0.008, 0.007, 0.006, 0.005]),
            err_dn = np.array([0.177, 0.052, 0.024, 0.016, 0.012, 0.009, 0.007, 0.006, 0.005]),
        ),
    },
    'galcut15': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([33.564, 4.943, 2.612, 1.691, 1.118, 0.623, 0.461, 0.363, 0.288]),
            err_up = np.array([0.235, 0.069, 0.033, 0.020, 0.014, 0.011, 0.009, 0.007, 0.007]),
            err_dn = np.array([0.242, 0.071, 0.034, 0.021, 0.014, 0.011, 0.009, 0.007, 0.007]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([13.185, 4.407, 2.148, 1.119, 0.688, 0.393, 0.308, 0.237, 0.190]),
            err_up = np.array([0.145, 0.044, 0.019, 0.014, 0.009, 0.007, 0.005, 0.006, 0.005]),
            err_dn = np.array([0.146, 0.046, 0.019, 0.014, 0.009, 0.007, 0.005, 0.005, 0.005]),
        ),
    },
    'galcut20': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([20.685, 4.137, 1.624, 1.178, 0.820, 0.436, 0.325, 0.265, 0.189]),
            err_up = np.array([0.229, 0.062, 0.029, 0.018, 0.014, 0.010, 0.009, 0.007, 0.006]),
            err_dn = np.array([0.224, 0.061, 0.029, 0.018, 0.014, 0.010, 0.009, 0.007, 0.006]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([8.333, 2.945, 1.345, 0.763, 0.461, 0.278, 0.235, 0.173, 0.137]),
            err_up = np.array([0.113, 0.036, 0.017, 0.012, 0.008, 0.006, 0.005, 0.005, 0.005]),
            err_dn = np.array([0.113, 0.037, 0.019, 0.012, 0.008, 0.006, 0.005, 0.005, 0.005]),
        ),
    },
}

# ---------------------------------------------------------------
# Figure 2: A_s^BB / A_s^EE  (single panel)
# ---------------------------------------------------------------
fig_r, ax_r = plt.subplots(1, 1, figsize=(15, 3.5))

for mask_key, mstyle in MASK_STYLES.items():
    ee   = DATA[mask_key]['EE']
    bb   = DATA[mask_key]['BB']
    xpos = ee['ell'] + mstyle['offset']

    ratio = bb['val'] / ee['val']

    # Error propagation for r = BB/EE:
    # sigma_r^± / r = sqrt( (sigma_BB^± / BB)^2 + (sigma_EE^± / EE)^2 )
    rel_up = np.sqrt((bb['err_up'] / bb['val'])**2 + (ee['err_up'] / ee['val'])**2)
    rel_dn = np.sqrt((bb['err_dn'] / bb['val'])**2 + (ee['err_dn'] / ee['val'])**2)

    ax_r.errorbar(
        xpos, ratio,
        yerr=[ratio * rel_dn, ratio * rel_up],
        fmt=mstyle['marker'],
        color=mstyle['color'],
        ms=6, capsize=3, capthick=1.2, lw=1.2, zorder=3,
        label=mstyle['label'],
    )

ax_r.axhline(0.2, color='grey', lw=0.8, ls='--', zorder=1)
ax_r.set_xlim(10, 220)
ax_r.set_ylim(0, 0.8)
ax_r.set_ylabel(r'$A^{\rm BB}_{\rm s}\,/\,A^{\rm EE}_{\rm s}$', fontsize=15)
ax_r.set_xlabel(r'$\ell$', fontsize=15)
ax_r.set_xticks(ell_ticks)
ax_r.set_xticklabels([str(int(v)) for v in ell_ticks])
ax_r.legend(
    loc='upper right', frameon=False, fontsize=15,
    handletextpad=0.4, borderpad=0.6,
)

plt.tight_layout()
plt.show()

# Save PDF
# fig_r.savefig(os.path.join(out_dir, 'As_ratio_BB_over_EE_bin_to_bin.pdf'), bbox_inches='tight')
print('Saved As_ratio_BB_over_EE_bin_to_bin.pdf')

# Transparent PNG
# save_transparent_white(
#     fig_r,
#     os.path.join(ppt_dir, 'As_ratio_BB_over_EE_bin_to_bin_transparent.png'),
#     dpi=300)

# ---------------------------------------------------------------
# Figure 3: amplitude ratios BB/EE — synch (top) + dust (bottom)
# ---------------------------------------------------------------
fig_r2, (ax_rs, ax_rd) = plt.subplots(
    2, 1, figsize=(15, 7), sharex=False,
    gridspec_kw={'hspace': 0.4})

for mask_key, mstyle in MASK_STYLES.items():
    xpos = DATA[mask_key]['EE']['ell'] + mstyle['offset']

    # --- synchrotron ratio ---
    ee_s = DATA[mask_key]['EE']
    bb_s = DATA[mask_key]['BB']
    ratio_s = bb_s['val'] / ee_s['val']
    rel_up_s = np.sqrt((bb_s['err_up'] / bb_s['val'])**2 + (ee_s['err_up'] / ee_s['val'])**2)
    rel_dn_s = np.sqrt((bb_s['err_dn'] / bb_s['val'])**2 + (ee_s['err_dn'] / ee_s['val'])**2)
    ax_rs.errorbar(
        xpos, ratio_s,
        yerr=[ratio_s * rel_dn_s, ratio_s * rel_up_s],
        fmt=mstyle['marker'], color=mstyle['color'],
        ms=6, capsize=3, capthick=1.2, lw=1.2, zorder=3,
        label=mstyle['label'],
    )

    # --- dust ratio ---
    ee_d = DATA_DUST[mask_key]['EE']
    bb_d = DATA_DUST[mask_key]['BB']
    ratio_d = bb_d['val'] / ee_d['val']
    rel_up_d = np.sqrt((bb_d['err_up'] / bb_d['val'])**2 + (ee_d['err_up'] / ee_d['val'])**2)
    rel_dn_d = np.sqrt((bb_d['err_dn'] / bb_d['val'])**2 + (ee_d['err_dn'] / ee_d['val'])**2)
    ax_rd.errorbar(
        xpos, ratio_d,
        yerr=[ratio_d * rel_dn_d, ratio_d * rel_up_d],
        fmt=mstyle['marker'], color=mstyle['color'],
        ms=6, capsize=3, capthick=1.2, lw=1.2, zorder=3,
        label=mstyle['label'],
    )

# ax_rs.axhline(0.25, color='grey', lw=0.8, ls='--', zorder=1)
ax_rs.set_xlim(10, 220)
ax_rs.set_ylim(0, 0.8)

# ax_rd.axhline(0.70, color='grey', lw=0.8, ls='--', zorder=1)
ax_rd.set_xlim(10, 220)
ax_rd.set_ylim(0.3, 1)

ax_rs.set_ylabel(r'$A^{\rm BB}_{\rm s}\,/\,A^{\rm EE}_{\rm s}$', fontsize=15)
ax_rd.set_ylabel(r'$A^{\rm BB}_{\rm d}\,/\,A^{\rm EE}_{\rm d}$', fontsize=15)

ax_rs.axhline(0.224, color='steelblue', lw=0.8, ls='--', zorder=1)
ax_rs.axhline(0.228, color='w', lw=0.8, ls='--', zorder=1)
ax_rs.axhline(0.229, color='goldenrod', lw=0.8, ls='--', zorder=1)

ax_rd.axhline(0.734, color='steelblue', lw=0.8, ls='--', zorder=1)
ax_rd.axhline(0.659, color='w', lw=0.8, ls='--', zorder=1)
ax_rd.axhline(0.644, color='goldenrod', lw=0.8, ls='--', zorder=1)

ax_rs.legend(
    loc='upper right', frameon=False, fontsize=15,
    handletextpad=0.4, borderpad=0.6,
)

ax_rd.legend(
    loc='upper right', frameon=False, fontsize=15,
    handletextpad=0.4, borderpad=0.6,
)

ax_rd.set_xticks(ell_ticks)
ax_rs.set_xticks(ell_ticks)
ax_rd.set_xticklabels([str(int(v)) for v in ell_ticks])
ax_rs.set_xticklabels([str(int(v)) for v in ell_ticks])
ax_rd.set_xlabel(r'$\ell$', fontsize=15)
ax_rs.set_xlabel(r'$\ell$', fontsize=15)

plt.tight_layout()
plt.show()

# Save PDF
fig_r2.savefig(os.path.join(out_dir, 'A_ratio_BB_over_EE_synch_dust_bin_to_bin.pdf'), bbox_inches='tight')
print('Saved A_ratio_BB_over_EE_synch_dust_bin_to_bin.pdf')

# Transparent PNG
save_transparent_white(
    fig_r2,
    os.path.join(ppt_dir, 'A_ratio_BB_over_EE_synch_dust_bin_to_bin_transparent_pl_values.png'),
    dpi=300)
