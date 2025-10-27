#%%
import re
import pandas as pd
import requests
from urllib.parse import quote
from pathlib import Path
from data import path_map
# === CONFIGURATION ===
base_dir = Path(path_map + "Planck/noise_simulations/")

# Use any subset of Planck frequency bands
bands = ["044", "070", "100", "143", "217", "353"]

# Base URL for the Planck Legacy Archive
BASE_URL = "https://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID="

# Define file patterns per band
file_patterns = {
    "030": r"ffp10_noise_030_full_map_mc_\d{5}\.fits$",
    "044": r"ffp10_noise_044_full_map_mc_\d{5}\.fits$",
    "070": r"ffp10_noise_070_full_map_mc_\d{5}\.fits$",
    "100": r"ffp10_noise_100_full_map_mc_\d{5}\.fits$",
    "143": r"ffp10_noise_143_full_map_mc_\d{5}\.fits$",
    "217": r"ffp10_noise_217_full_map_mc_\d{5}\.fits$",
    "353": r"ffp10_noise_353_psb_full_map_mc_\d{5}\.fits$",
}

# === LOOP OVER BANDS ===
for band in bands:
    band_int = str(int(band))  # Remove leading zeros for folder
    band_dir = base_dir / f"{band_int}_pla"
    csv_path = band_dir / "maps.csv"

    if not csv_path.exists():
        print(f"CSV not found for {band_int} GHz at {csv_path}")
        continue

    print(f"\n=== Processing {band_int} GHz ===")

    # Read CSV
    df = pd.read_csv(csv_path, dtype=str)

    # Detect file column
    file_cols = [c for c in df.columns if 'file' in c.lower()]
    if not file_cols:
        print(f"No column containing 'file' found in {csv_path}")
        continue
    col = file_cols[0]

    # Filter files using the band-specific pattern
    pattern = re.compile(file_patterns[band], re.IGNORECASE)
    mask = df[col].fillna("").str.contains(pattern)
    df_filtered = df[mask].copy()

    print(f"{len(df_filtered)} files matched for {band_int} GHz")

    if len(df_filtered) == 0:
        continue

    # Print first and last file for sanity check
    print("First file:", df_filtered[col].iloc[0])
    print("Last file:", df_filtered[col].iloc[-1])

    # Download each FITS file
    for fname in df_filtered[col]:
        base_name = Path(fname).name
        url = BASE_URL + quote(base_name)
        dest = band_dir / base_name

        if dest.exists():
            print(f"Already exists: {base_name}")
            continue

        print(f"Downloading {base_name} ...")
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                dest.write_bytes(r.content)
            else:
                print(f"Error {r.status_code} downloading {base_name}")
        except Exception as e:
            print(f"Failed {base_name}: {e}")

print("\nAll bands processed.")
print(f"Base directory: {base_dir.resolve()}")
