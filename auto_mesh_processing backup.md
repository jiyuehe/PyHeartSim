---
marp: true
theme: default
paginate: true
size: 16:9
title: Atria TSDF Rebuild Pipeline
---

# Atria TSDF Rebuild Pipeline

Refactor + usage guide + parameter behavior

`main.py` now supports both:
- CLI: `python main.py ...`
- API: `TSDFRebuilder(TSDFConfig(...)).run(...)`

---

# Project structure

```
atria_tsdf/
  main.py
  raw_mesh/
  presentation/
    tsdf_pipeline.md
    figures/
    outputs/
```

- `main.py` contains config, pipeline class, CLI parser, and entrypoint.
- `presentation/figures` contains charts and static visuals.
- `presentation/outputs` contains parameter-sweep output meshes.

---

# How the pipeline works

1. Load and validate mesh (`trimesh`).
2. Voxelize at selected target resolution.
3. Morphological operations (dilation/closing/fill holes).
4. Build signed distance field (SDF) and smooth it.
5. Marching cubes reconstruction.
6. Optional decimation, smoothing, isotropic remesh.
7. Export OBJ + metrics report.

---

# Core implementation layout (`main.py`)

- `TSDFConfig`: all tunable parameters and defaults.
- `TSDFRebuilder`:
  - `run(input_path, output_path)` for end-to-end rebuild
  - `_maybe_visualize(...)` for 3-panel debug visualization
- Utilities:
  - `decimate_mesh_with_best_api`
  - `laplacian_smooth`
  - `crinkliness_metric_np`
- CLI:
  - `build_parser()`, `config_from_args()`, `main()`

---

# CLI usage

```bash
python main.py --help

python main.py \
  --input raw_mesh/105_6-LA.obj \
  --output tsdf_test.obj \
  --target-res 120 \
  --sdf-smoothing-sigma 2.0 \
  --smooth-iterations 10
```

Key toggles:
- `--disable-remesh`
- `--disable-decimation`
- `--visualize`

---

# Class/API usage

```python
from main import TSDFConfig, TSDFRebuilder

cfg = TSDFConfig(
    tsdf_target_res=120,
    sdf_smoothing_sigma=2.0,
    enable_remesh=False,
    visualize=False,
)
report = TSDFRebuilder(cfg).run("raw_mesh/105_6-LA.obj", "out.obj")
print(report["status"], report["output_path"])
```

This mirrors CLI behavior while allowing programmatic integration.

---

# Dependency behavior (explicit failures)

All required libraries are hard imports:
- `trimesh`
- `scipy`
- `scikit-image`
- `pymeshlab`
- `pyvista`

If any are missing, startup fails immediately with `ModuleNotFoundError`.

---

# Figure: pipeline stages

![w:1200](figures/pipeline_steps.png)

Raw input → marching cubes surface → final smoothed output.

---

# Figure: target resolution vs reconstructed faces

![w:1100](figures/fig_target_res_vs_faces.svg)

Higher `tsdf_target_res` generally increases reconstructed triangle count.

---

# Figure: SDF sigma vs crinkliness

![w:1100](figures/fig_sigma_vs_crinkliness.svg)

Larger `sdf_smoothing_sigma` reduces high-frequency surface crinkliness.

---

# Parameter sweep data snapshot

Source: `presentation/figures/parameter_sweep.json`

- `target_res`: 80, 120, 160
- `sdf_sigma`: 0.5, 2.0, 4.0
- Metrics tracked:
  - `reconstructed_faces`
  - `crinkliness_rec_mean_deg`
  - output size

---

# Practical tuning guidance

- Start with:
  - `--target-res 120`
  - `--sdf-smoothing-sigma 2.0`
  - `--smooth-iterations 10`
- Need smoother mesh:
  - increase `sdf_smoothing_sigma`
  - increase smoothing iterations
- Need more detail:
  - raise `target-res`
  - lower `sdf_smoothing_sigma`

---

# Summary

- `main.py` was refactored for CLI + class invocation.
- Dependency fallbacks were removed for explicit startup failures.
- Presentation assets document structure, flow, usage, and parameter effects.

Artifacts are under `presentation/`.
