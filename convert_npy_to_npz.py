"""
Convert simulation_results_*.npy (pickled dicts) and lat_*.npy (plain arrays) to .npz files.
Run this script in the environment where the .npy files were created (newer NumPy).
Usage: python convert_npy_to_npz.py <folder>
"""
import sys
import numpy as np
from pathlib import Path


def convert_folder(folder: Path):
    # --- simulation_results_*.npy (pickled dicts) ---
    npy_files = list(folder.glob('simulation_results_*.npy'))
    if not npy_files:
        print(f'No simulation_results_*.npy files found in {folder}')
    else:
        print(f'Found {len(npy_files)} simulation_results files in {folder}')
        for i, f in enumerate(npy_files, 1):
            print(f'  [{i}/{len(npy_files)}] {f.name} ...', end=' ', flush=True)
            npz_path = f.with_suffix('.npz')
            if npz_path.exists():
                print('SKIP (already exists)')
                continue
            try:
                data = np.load(f, allow_pickle=True).item()
                # Convert all values to numpy arrays where possible
                save_kwargs = {}
                for key, value in data.items():
                    try:
                        save_kwargs[key] = np.asarray(value)
                    except Exception:
                        save_kwargs[key] = value
                np.savez(npz_path, **save_kwargs)
                f.unlink()
                print('OK')
            except Exception as e:
                print(f'ERROR: {e}')

    # # --- lat_*.npy (plain arrays) ---
    # lat_files = list(folder.glob('lat_*.npy'))
    # if not lat_files:
    #     print(f'No lat_*.npy files found in {folder}')
    # else:
    #     print(f'Found {len(lat_files)} lat files in {folder}')
    #     for i, f in enumerate(lat_files, 1):
    #         print(f'  [{i}/{len(lat_files)}] {f.name} ...', end=' ', flush=True)
    #         npz_path = f.with_suffix('.npz')
    #         if npz_path.exists():
    #             print('SKIP (already exists)')
    #             continue
    #         try:
    #             lat = np.load(f, allow_pickle=False)
    #             np.savez(npz_path, lat=lat)
    #             f.unlink()
    #             print('OK')
    #         except Exception as e:
    #             print(f'ERROR: {e}')


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print('Usage: python convert_npy_to_npz.py <folder> [<folder2> ...]')
    #     sys.exit(1)
    # for arg in sys.argv[1:]:
    convert_folder(Path('/home/j/Desktop/hdd/103_1-lagood/npy'))
