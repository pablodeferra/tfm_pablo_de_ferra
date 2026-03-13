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
# ρ data extracted from the three bin-to-bin tables (QJ+WP+Pl)
# Structure: RHO[mask_key][mode] = dict(ell, rho, err_up, err_dn)
# ---------------------------------------------------------------
RHO = {
    'galcut10': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            rho    = np.array([0.092, 0.117, 0.044, 0.055, 0.037, 0.174, 0.225, 0.073, 0.119]),
            err_up = np.array([0.007, 0.013, 0.014, 0.021, 0.028, 0.044, 0.057, 0.062, 0.099]),
            err_dn = np.array([0.007, 0.014, 0.014, 0.019, 0.027, 0.041, 0.048, 0.059, 0.068]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            rho    = np.array([0.117,  0.104, 0.088, 0.153, 0.104,  0.143, -0.029, -0.054, 0.068]),
            err_up = np.array([0.014,  0.020, 0.055, 0.096, 0.071,  0.062,  0.141,  0.051, 0.079]),
            err_dn = np.array([0.014,  0.020, 0.042, 0.055, 0.049,  0.044,  0.154,  0.056, 0.065]),
        ),
    },
    'galcut15': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            rho    = np.array([0.078,  0.046, -0.007, 0.080, 0.097, 0.123, 0.148, 0.030, 0.111]),
            err_up = np.array([0.008,  0.016,  0.019, 0.023, 0.030, 0.049, 0.058, 0.063, 0.137]),
            err_dn = np.array([0.008,  0.016,  0.018, 0.022, 0.029, 0.044, 0.050, 0.059, 0.085]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            rho    = np.array([0.164,  0.041, 0.121, 0.125,  0.041,  0.091,  0.023, -0.097, -0.004]),
            err_up = np.array([0.017,  0.022, 0.091, 0.093,  0.060,  0.066,  0.163,  0.061,  0.063]),
            err_dn = np.array([0.017,  0.021, 0.057, 0.062,  0.053,  0.055,  0.135,  0.068,  0.061]),
        ),
    },
    'galcut20': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            rho    = np.array([0.009,  0.067, -0.091, 0.062, 0.041, 0.180, 0.097, 0.027, 0.122]),
            err_up = np.array([0.010,  0.015,  0.022, 0.025, 0.038, 0.059, 0.052, 0.064, 0.145]),
            err_dn = np.array([0.009,  0.015,  0.022, 0.024, 0.036, 0.052, 0.047, 0.062, 0.099]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            rho    = np.array([0.171,  0.035,  0.185, 0.079, 0.046,  0.106, -0.004, -0.082, -0.077]),
            err_up = np.array([0.020,  0.028,  0.121, 0.094, 0.064,  0.081,  0.155,  0.062,  0.064]),
            err_dn = np.array([0.020,  0.027,  0.072, 0.070, 0.059,  0.064,  0.144,  0.067,  0.067]),
        ),
    },
}

# ---------------------------------------------------------------
# Visual encoding — mirrors the reference figure
# mask shapes, offset along ell for clarity
# ---------------------------------------------------------------
MASK_STYLES = {
    'galcut10': dict(color='steelblue',  marker='o', label=r'$\pm10\degree$', offset=-1.7),
    'galcut15': dict(color='k', marker='^', label=r'$\pm15\degree$', offset= 0),
    'galcut20': dict(color='goldenrod',  marker='s', label=r'$\pm20\degree$', offset=+1.7),
}

# ---------------------------------------------------------------
# Figure: 2 stacked panels (EE top, BB bottom), shared x-axis
# ---------------------------------------------------------------
fig, (ax_EE, ax_BB) = plt.subplots(
    2, 1, figsize=(15, 6), sharex=True,
    gridspec_kw={'hspace': 0.15})

ell_ticks = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float)

# (legend will be created from the plotted errorbars so markers and caplines
# appear correctly in the legend)

for ax, mode, panel_label in [
        (ax_EE, 'EE', 'EE'),
        (ax_BB, 'BB', 'BB')]:

    # Panel titles removed; using y-axis labels to indicate EE/BB
    ax.axhline(0.1, color='grey', lw=0.8, ls='--', zorder=1)

    for mask_key, mstyle in MASK_STYLES.items():
        d    = RHO[mask_key][mode]
        xpos = d['ell'] + mstyle['offset']
        ax.errorbar(
            xpos,
            d['rho'],
            yerr=[d['err_dn'], d['err_up']],
            fmt=mstyle['marker'],
            color=mstyle['color'],
            ms=6,
            capsize=3,
            capthick=1.2,
            lw=1.2,
            zorder=3,
            label=mstyle['label'] if mode == 'EE' else None,
        )

    # Label y-axis with rho^{EE} or rho^{BB} instead of placing titles at the top
    if mode == 'EE':
        ax.set_ylabel(r'$\rho^{EE}$', fontsize=15)
    else:
        ax.set_ylabel(r'$\rho^{BB}$', fontsize=15)
    # ax.grid(True, axis='y', lw=0.4, alpha=0.5)
    ax.set_xlim(10, 220)
    ax.set_ylim(-0.2, 0.4)

# legend inside the EE panel (top right)
ax_EE.legend(
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
fig.savefig(os.path.join(out_dir, 'rho_vs_ell_bin_to_bin.pdf'), bbox_inches='tight')
print('Saved rho_vs_ell_bin_to_bin.pdf')

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
    os.path.join(ppt_dir, 'rho_vs_ell_bin_to_bin_transparent.png'),
    dpi=300)

plt.show()