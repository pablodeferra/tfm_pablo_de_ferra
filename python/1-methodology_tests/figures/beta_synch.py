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
# beta_s data extracted from the three bin-to-bin tables (QJ+WP+Pl)
# Structure: DATA[mask_key][mode] = dict(ell, val, err_up, err_dn)
# ---------------------------------------------------------------
DATA = {
    'galcut10': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([-3.075, -3.135, -3.234, -3.133, -2.966, -2.594, -2.909, -3.094, -2.903]),
            err_up = np.array([0.018, 0.050, 0.068, 0.116, 0.165, 0.230, 0.238, 0.278, 0.296]),
            err_dn = np.array([0.018, 0.047, 0.069, 0.110, 0.164, 0.219, 0.238, 0.271, 0.299]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([-3.010, -3.125, -3.154, -2.988, -2.697, -2.965, -2.976, -3.070, -3.169]),
            err_up = np.array([0.059, 0.112, 0.272, 0.232, 0.246, 0.247, 0.333, 0.319, 0.326]),
            err_dn = np.array([0.059, 0.110, 0.254, 0.239, 0.259, 0.237, 0.309, 0.310, 0.320]),
        ),
    },
    'galcut15': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([-3.078, -3.107, -3.234, -3.148, -2.907, -2.815, -2.822, -2.991, -2.909]),
            err_up = np.array([0.019, 0.056, 0.084, 0.120, 0.170, 0.247, 0.259, 0.280, 0.305]),
            err_dn = np.array([0.019, 0.056, 0.083, 0.119, 0.170, 0.243, 0.273, 0.273, 0.305]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([-3.086, -3.092, -3.118, -2.812, -2.754, -3.018, -2.968, -3.229, -2.877]),
            err_up = np.array([0.070, 0.127, 0.261, 0.241, 0.269, 0.271, 0.305, 0.277, 0.249]),
            err_dn = np.array([0.068, 0.127, 0.262, 0.258, 0.290, 0.270, 0.309, 0.278, 0.271]),
        ),
    },
    'galcut20': {
        'EE': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([-3.103, -3.119, -3.210, -3.144, -2.950, -2.783, -2.672, -3.016, -2.911]),
            err_up = np.array([0.021, 0.056, 0.085, 0.128, 0.200, 0.237, 0.219, 0.279, 0.299]),
            err_dn = np.array([0.021, 0.055, 0.085, 0.130, 0.205, 0.242, 0.236, 0.274, 0.300]),
        ),
        'BB': dict(
            ell    = np.array([29, 49, 69, 89, 109, 129, 149, 169, 189], dtype=float),
            val    = np.array([-3.039, -3.086, -2.963, -2.872, -2.750, -3.033, -2.954, -3.149, -2.741]),
            err_up = np.array([0.072, 0.142, 0.289, 0.254, 0.280, 0.268, 0.308, 0.269, 0.218]),
            err_dn = np.array([0.072, 0.134, 0.288, 0.267, 0.290, 0.263, 0.308, 0.269, 0.245]),
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

for ax, mode in [(ax_EE, 'EE'), (ax_BB, 'BB')]:

    ax.axhline(-3.1, color='grey', lw=0.8, ls='--', zorder=1)

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
            label=mstyle['label'] if mode == 'EE' else None,
        )

    if mode == 'EE':
        ax.set_ylabel(r'$\beta^{\rm EE}_{\rm s}$', fontsize=15)
    else:
        ax.set_ylabel(r'$\beta^{\rm BB}_{\rm s}$', fontsize=15)
    ax.set_xlim(10, 220)
    ax.set_ylim(-3.6, -2.4)

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
fig.savefig(os.path.join(out_dir, 'beta_s_vs_ell_bin_to_bin.pdf'), bbox_inches='tight')
print('Saved beta_s_vs_ell_bin_to_bin.pdf')

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
    os.path.join(ppt_dir, 'beta_s_vs_ell_bin_to_bin_transparent.png'),
    dpi=300)

plt.show()
