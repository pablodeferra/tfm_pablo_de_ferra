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
    'galcut15': dict(color='k',          marker='^', label=r'$\pm15\degree$', offset= 0),
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
fig.savefig(os.path.join(out_dir, 'As_vs_ell_bin_to_bin.pdf'), bbox_inches='tight')
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
save_transparent_white(
    fig,
    os.path.join(ppt_dir, 'As_vs_ell_bin_to_bin_transparent.png'),
    dpi=300)

plt.show()
