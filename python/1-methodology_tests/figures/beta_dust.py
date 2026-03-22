#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
# beta_d data extracted from the three bin-to-bin tables (QJ+WP+Pl)
# Structure: DATA[mask_key][mode] = dict(ell, val, err_up, err_dn)
# ---------------------------------------------------------------
DATA = {
    'galcut10': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([1.481, 1.519, 1.531, 1.504, 1.567, 1.420, 1.477, 1.543, 1.635]),
            err_up = np.array([0.007, 0.014, 0.012, 0.013, 0.019, 0.021, 0.026, 0.029, 0.032]),
            err_dn = np.array([0.007, 0.013, 0.012, 0.013, 0.018, 0.020, 0.025, 0.028, 0.032]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([1.509, 1.493, 1.550, 1.552, 1.572, 1.540, 1.514, 1.533, 1.488]),
            err_up = np.array([0.010, 0.012, 0.012, 0.013, 0.015, 0.018, 0.023, 0.025, 0.024]),
            err_dn = np.array([0.010, 0.011, 0.011, 0.013, 0.015, 0.017, 0.022, 0.024, 0.023]),
        ),
    },
    'galcut15': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([1.467, 1.584, 1.514, 1.519, 1.552, 1.380, 1.484, 1.568, 1.656]),
            err_up = np.array([0.009, 0.020, 0.019, 0.019, 0.021, 0.029, 0.033, 0.035, 0.045]),
            err_dn = np.array([0.010, 0.019, 0.019, 0.018, 0.022, 0.027, 0.033, 0.034, 0.046]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([1.484, 1.494, 1.571, 1.560, 1.577, 1.557, 1.521, 1.510, 1.512]),
            err_up = np.array([0.014, 0.014, 0.015, 0.020, 0.024, 0.034, 0.033, 0.042, 0.044]),
            err_dn = np.array([0.014, 0.014, 0.015, 0.020, 0.024, 0.033, 0.032, 0.041, 0.042]),
        ),
    },
    'galcut20': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([1.472, 1.508, 1.446, 1.485, 1.564, 1.320, 1.450, 1.543, 1.625]),
            err_up = np.array([0.013, 0.020, 0.025, 0.024, 0.030, 0.036, 0.046, 0.047, 0.069]),
            err_dn = np.array([0.014, 0.020, 0.025, 0.024, 0.029, 0.035, 0.045, 0.045, 0.066]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([1.511, 1.480, 1.607, 1.579, 1.527, 1.587, 1.556, 1.513, 1.519]),
            err_up = np.array([0.020, 0.019, 0.024, 0.028, 0.032, 0.044, 0.042, 0.054, 0.061]),
            err_dn = np.array([0.019, 0.018, 0.023, 0.029, 0.030, 0.044, 0.041, 0.054, 0.060]),
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
    2, 1, figsize=(15, 9), sharex=False,
    gridspec_kw={'hspace': 0.35})

ell_ticks = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float)

# (legend will be created from the plotted errorbars so markers and caplines
# appear correctly in the legend)

for ax, mode in [(ax_EE, 'EE'), (ax_BB, 'BB')]:

    # ax.axhline(1.53, color='grey', lw=0.8, ls='--', zorder=1)

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
            label=mstyle['label']
        )

    if mode == 'EE':
        ax.set_ylabel(r'$\beta^{\rm EE}_{\rm d}$', fontsize=15)
    else:
        ax.set_ylabel(r'$\beta^{\rm BB}_{\rm d}$', fontsize=15)
    ax.set_xlim(20, 200)
    ax.set_ylim(1.53-0.3, 1.53+0.2)

# legend inside the EE panel (upper right)
ax_EE.legend(
    loc='lower right',
    frameon=False,
    framealpha=0.85,
    edgecolor='grey',
    fontsize=15,
    handletextpad=0.4,
    borderpad=0.6,
)

ax_BB.legend(
    loc='lower right',
    frameon=False,
    framealpha=0.85,
    edgecolor='grey',
    fontsize=15,
    handletextpad=0.4,
    borderpad=0.6,
)

ax_EE.axhline(1.530, color='steelblue', lw=0.8, ls='--', zorder=1)
ax_EE.axhline(1.551, color='w', lw=0.8, ls='--', zorder=1)
ax_EE.axhline(1.486, color='goldenrod', lw=0.8, ls='--', zorder=1)

ax_BB.axhline(1.530, color='steelblue', lw=0.8, ls='--', zorder=1)
ax_BB.axhline(1.531, color='w', lw=0.8, ls='--', zorder=1)
ax_BB.axhline(1.533, color='goldenrod', lw=0.8, ls='--', zorder=1)



ax_BB.set_xticks(ell_ticks)
ax_BB.set_xticklabels([str(int(v)) for v in ell_ticks])
ax_BB.set_xlabel(r'$\ell$', fontsize=15)
ax_EE.set_xlabel(r'$\ell$', fontsize=15)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Save PDF
# ---------------------------------------------------------------
out_dir = '/home/pablo/Desktop/master/tfm/figures/spectra/'
os.makedirs(out_dir, exist_ok=True)
# fig.savefig(os.path.join(out_dir, 'beta_d_vs_ell_bin_to_bin.pdf'), bbox_inches='tight')
print('Saved beta_d_vs_ell_bin_to_bin.pdf')

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
save_transparent_white(
    fig,
    # os.path.join(ppt_dir, 'beta_d_vs_ell_bin_to_bin_transparent.png'),
    os.path.join(ppt_dir, 'beta_d_vs_ell_bin_to_bin_transparent_pl_values.png'),
    dpi=300)

plt.show()
