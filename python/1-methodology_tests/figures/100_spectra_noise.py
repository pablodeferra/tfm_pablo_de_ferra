#%%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from pathlib import Path
from typing import Optional

# Plot style
plt.rcParams.update({
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.titlecolor': 'white',
    'legend.facecolor': 'black',
    'legend.edgecolor': 'white',
    'legend.fontsize': 'medium',
    'text.color': 'white',
    'figure.facecolor': 'black',
    'figure.edgecolor': 'white',
    'axes.facecolor': 'black',
    'axes.edgecolor': 'white',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,  
})


def check_file_path(base_path):
    """Check if file exists, otherwise try .gz"""
    if os.path.exists(base_path):
        return base_path
    if os.path.exists(base_path + ".gz"):
        return base_path + ".gz"
    return None


def resolve_full_sims_path(path_full: str) -> Optional[str]:
    """Resolve the full-sims file path.

    In this repo there are multiple naming conventions:
    - spectra_full_quijote_<mask>_<n>_<type>_<binning>.fits(.gz)
    - spectra_full_<mask>_<n>_<type>_<binning>.fits.gz

    We prefer the compressed file when available.
    """
    if path_full is None:
        return None

    # If caller provided a .fits path, prefer a sibling .fits.gz if it exists.
    if path_full.endswith(".fits") and os.path.exists(path_full + ".gz"):
        return path_full + ".gz"

    # Standard check (.fits or .fits.gz)
    resolved = check_file_path(path_full)
    if resolved is not None:
        return resolved

    # Fallback: try the alternate naming where "quijote_" is dropped.
    alt = path_full.replace("spectra_full_quijote_", "spectra_full_")
    if alt != path_full:
        # Prefer compressed
        if alt.endswith(".fits") and os.path.exists(alt + ".gz"):
            return alt + ".gz"
        resolved = check_file_path(alt)
        if resolved is not None:
            return resolved

    return None


def read_mode_sims_from_full(file_full: str, band_key: str, mode: str):
    """Read per-simulation spectra array from the full-sims FITS.

    The full files in this project typically store each mode in a dedicated HDU:
    e.g. '11_11__EE__SIMS_SIMS' with columns SIM_1..SIM_N.

    Returns:
      ell (1d), sims (2d: n_ell x n_sims)
    """
    hdu_name = f"{band_key}__{mode}__SIMS_SIMS"
    with fits.open(file_full) as hdul:
        if hdu_name not in hdul:
            raise KeyError(f"HDU '{hdu_name}' not found")
        d = hdul[hdu_name].data
        ell = d["ELL_EFF"]
        sim_cols = [c for c in d.columns.names if c.startswith("SIM_")]
        if len(sim_cols) == 0:
            raise ValueError(f"No SIM_* columns found in HDU '{hdu_name}'")
        sims = np.vstack([d[c] for c in sim_cols]).T  # (n_ell, n_sims)
    return ell, sims


def plot_spectra(file_full, file_stats, output_path, title, band_key):

    print(f"\nProcessing {title}")
    print(f"Band: {band_key}")

    file_full = resolve_full_sims_path(file_full)
    file_stats = check_file_path(file_stats)

    if file_stats is None:
        print("Stats file not found.")
        return

    # -------------------
    # LOAD STATS FILE
    # -------------------
    with fits.open(file_stats) as hdul:

        print("Stats extensions:", [h.name for h in hdul])

        try:
            data_stats = hdul[band_key].data
        except KeyError:
            print(f"Extension {band_key} not found in stats file.")
            return

        ell = data_stats["ell_eff_MEAN"]

        cl_ee_mean = data_stats["EE_MEAN"]
        cl_bb_mean = data_stats["BB_MEAN"]

        cl_ee_std = data_stats["EE_STD"]
        cl_bb_std = data_stats["BB_STD"]

    # -------------------
    # PLOTTING
    # -------------------

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    modes = ["EE", "BB"]
    means = [cl_ee_mean, cl_bb_mean]
    stds = [cl_ee_std, cl_bb_std]
    # We intentionally plot ONLY the mean/std envelopes from the avg/std products,
    # since those are the quantities actually used as error bars in the notebook.

    for i, mode in enumerate(modes):

        ax = axes[i]

        # -------------------
        # Plot simulations
        # -------------------

        # Mean +/- 1 sigma used for error bars
        ax.fill_between(
            ell,
            means[i] - stds[i],
            means[i] + stds[i],
            color="white",
            alpha=0.30,
            linewidth=0,
            zorder=1,
            label=r"$\pm 1\sigma$",
        )

        # Optional wider band for visual reference
        ax.fill_between(
            ell,
            means[i] - 2 * stds[i],
            means[i] + 2 * stds[i],
            color="white",
            alpha=0.12,
            linewidth=0,
            zorder=0,
            label=r"$\pm 2\sigma$",
        )

        # -------------------
        # Plot mean
        # -------------------

        # Mean curve (used as the central value)
        ax.plot(
            ell,
            means[i],
            color="cyan",
            linewidth=2,
            label="Mean",
            zorder=2,
        )

        ax.set_title(f"{mode} Mode", fontsize=18)
        ax.set_xlabel(r"$\ell$", fontsize=18)

        if i == 0:
            ax.set_ylabel(r"$N_\ell \ [\rm mK^2]$", fontsize=18)

    ax.set_yscale("log")
        # ax.set_ylim(6e-9, 1e-7)


    # fig.suptitle(title, fontsize=20)

    # Put legend only once to keep the figure clean
    axes[-1].legend(loc="upper right", frameon=False, fontsize=18)

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, transparent=True)

    print("Saved to:", output_path)

    plt.close(fig)


# -------------------------------------------------
# EXECUTION
# -------------------------------------------------

path_spectra = "/home/pablo/Desktop/master/tfm/spectra/"
mask_suffix = "galcut10"
n_sim = 100
binning = "full_bin_20-199"

output_dir = "/home/pablo/Desktop/master/tfm/figures_ppt/spectra_100"

# Files

full_sn = os.path.join(
    path_spectra,
    f"spectra_full_quijote_{mask_suffix}_{n_sim}_skyplusnoise_{binning}.fits",
)

avgstd_sn = os.path.join(
    path_spectra,
    f"spectra_avg_std_quijote_{mask_suffix}_avg_std{n_sim}_skyplusnoise_{binning}.fits",
)

full_n = os.path.join(
    path_spectra,
    f"spectra_full_quijote_{mask_suffix}_{n_sim}_noise_{binning}.fits",
)

avgstd_n = os.path.join(
    path_spectra,
    f"spectra_avg_std_quijote_{mask_suffix}_avg_std{n_sim}_noise_{binning}.fits",
)


# -------------------------
# PLANCK 30 GHz
# -------------------------

plot_spectra(
    full_sn,
    avgstd_sn,
    os.path.join(output_dir, f"spectra_30_skyplusnoise_{mask_suffix}.png"),
    f"Planck 30 GHz Sky+Noise ({n_sim} sims, {mask_suffix})",
    band_key="30_30",
)

plot_spectra(
    full_n,
    avgstd_n,
    os.path.join(output_dir, f"spectra_30_noise_{mask_suffix}.png"),
    f"Planck 30 GHz Noise ({n_sim} sims, {mask_suffix})",
    band_key="30_30",
)