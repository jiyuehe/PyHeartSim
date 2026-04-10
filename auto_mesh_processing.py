#%%
import math
import os
from typing import Optional

import numpy as np
import pymeshlab as pml # pip install pymeshlab
import pyvista as pv # pip install pyvista
import trimesh # pip install trimesh
from scipy import ndimage
from skimage import measure # pip install scikit-image
# pip install fast-simplification

import os
from pathlib import Path

# add the workspace root to Python path
import sys
workspace_root = Path().resolve().parent # Path().resolve() returns an absolute path, the full path
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
import common

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def clean_mesh(
    input_path: str,
    output_path: str,
    *,
    debug_mode: bool,
    tsdf_target_res: int,
    tsdf_truncation_dist: Optional[float],
    morph_closing_iters: int,
    morph_dilation_iters: int,
    pad_voxels: int,
    fill_internal_volume: bool,
    sdf_smoothing_sigma: float,
    mc_level: float,
    simplify_faces_ratio: float,
    enable_decimation: bool,
    smooth_iterations: int,
    smooth_lambda: float,
    enable_remesh: bool,
    target_edge_length: Optional[float],
    post_remesh_smooth_iterations: int,
    visualize: bool,
) -> dict:
    """Run the full mesh-cleaning pipeline and export a processed mesh.

    The function is intentionally linear: each numbered step performs one stage
    of the TSDF-based reconstruction workflow and records key values in report.
    """
    report = {
        "status": "starting",
        "input_used": input_path,
        "config": {
            "debug_mode": debug_mode,
            "tsdf_target_res": tsdf_target_res,
            "tsdf_truncation_dist": tsdf_truncation_dist,
            "morph_closing_iters": morph_closing_iters,
            "morph_dilation_iters": morph_dilation_iters,
            "pad_voxels": pad_voxels,
            "fill_internal_volume": fill_internal_volume,
            "sdf_smoothing_sigma": sdf_smoothing_sigma,
            "mc_level": mc_level,
            "simplify_faces_ratio": simplify_faces_ratio,
            "enable_decimation": enable_decimation,
            "smooth_iterations": smooth_iterations,
            "smooth_lambda": smooth_lambda,
            "enable_remesh": enable_remesh,
            "target_edge_length": target_edge_length,
            "post_remesh_smooth_iterations": post_remesh_smooth_iterations,
            "visualize": visualize,
        },
    }

    # 1) Load the input mesh from disk.
    mesh = trimesh.load(input_path, force="mesh")
    if mesh.is_empty:
        raise RuntimeError("Loaded mesh is empty.")

    # 2) If the mesh is not watertight, run a basic repair pass.
    #    Open boundaries often cause poor voxel occupancy and surface artifacts.
    if not mesh.is_watertight:
        if debug_mode:
            print("[DEBUG] Input mesh is not watertight, attempting repairs...")
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        mesh.fill_holes()

    # 3) Compute voxel pitch from mesh size and target resolution.
    #    Pitch controls reconstruction detail and TSDF grid memory usage.
    bbox = mesh.bounds
    extent = bbox[1] - bbox[0]
    diag = np.linalg.norm(extent)
    pitch = (
        float(max(extent) / tsdf_target_res)
        if max(extent) > 0
        else (diag / tsdf_target_res if diag > 0 else 0.001)
    )
    report["grid_pitch"] = float(pitch)
    if debug_mode:
        print(f"[DEBUG] Grid pitch set to: {pitch}")

    # 4) Convert the mesh surface into a binary voxel occupancy grid.
    #    TSDF generation and marching cubes operate on grid data.
    voxelized = mesh.voxelized(pitch)
    vox_matrix = voxelized.matrix.copy()
    report["vox_shape"] = vox_matrix.shape
    report["vox_filled_count"] = int(np.count_nonzero(vox_matrix))
    if debug_mode:
        print(
            f"[DEBUG] Initial voxelization complete. Voxel shape {vox_matrix.shape}, "
            f"filled voxels: {report['vox_filled_count']}"
        )

    # 5) Apply morphology to the occupancy grid.
    #    Padding avoids border clipping, dilation/closing bridge tiny gaps,
    #    and hole filling reduces disconnected or hollow artifacts after meshing.
    occ = vox_matrix.astype(bool)
    if pad_voxels > 0:
        occ = np.pad(occ, pad_width=pad_voxels, mode="constant", constant_values=False)
        if debug_mode:
            print(f"[DEBUG] Grid padded by {pad_voxels} voxels to prevent mesh border holes.")

    if morph_dilation_iters > 0:
        occ = ndimage.binary_dilation(occ, iterations=morph_dilation_iters)
        if debug_mode:
            print(
                f"[DEBUG] Morphological dilation applied (iters={morph_dilation_iters}). "
                f"New filled: {np.count_nonzero(occ)}"
            )

    if morph_closing_iters > 0:
        occ = ndimage.binary_closing(occ, iterations=morph_closing_iters)
        if debug_mode:
            print(
                f"[DEBUG] Morphological closing applied (iters={morph_closing_iters}). "
                f"New filled: {np.count_nonzero(occ)}"
            )

    if fill_internal_volume:
        occ = ndimage.binary_fill_holes(occ)
        if debug_mode:
            print(
                "[DEBUG] Binary Hole Filling applied to discard internal walls. "
                f"New filled voxels: {np.count_nonzero(occ)}"
            )

    # 6) Build signed distance field (SDF), with optional truncation and smoothing.
    #    The zero level-set of the SDF is the reconstructed surface.
    outside_dist = ndimage.distance_transform_edt(~occ) * pitch
    inside_dist = ndimage.distance_transform_edt(occ) * pitch
    sdf = outside_dist.copy()
    sdf[occ] = -inside_dist[occ]

    if tsdf_truncation_dist is not None:
        sdf = np.clip(sdf, -tsdf_truncation_dist, tsdf_truncation_dist)

    if sdf_smoothing_sigma > 0:
        sdf = ndimage.gaussian_filter(sdf, sigma=sdf_smoothing_sigma)
        if debug_mode:
            print(f"[DEBUG] Gaussian smoothing applied on SDF with sigma={sdf_smoothing_sigma}.")

    # 7) Extract an isosurface from the SDF using marching cubes.
    #    This converts the volumetric field back into a triangle mesh.
    origin = bbox[0] - (pitch / 2.0) - (pitch * pad_voxels)
    report["assumed_origin"] = origin.tolist()

    if debug_mode:
        print(f"[DEBUG] Running Marching Cubes at iso-level {mc_level}...")

    verts, faces, _, _ = measure.marching_cubes(
        sdf, level=mc_level, spacing=(pitch, pitch, pitch)
    )

    verts_world = verts + origin
    reconstructed = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    mc_mesh = reconstructed.copy()
    report["reconstructed_vertices"] = int(len(reconstructed.vertices))
    report["reconstructed_faces"] = int(len(reconstructed.faces))

    # 8) Optionally decimate and pre-smooth.
    #    Reduces unnecessary complexity and relaxes high-frequency noise.
    if enable_decimation:
        target_faces = min(
            len(reconstructed.faces), max(10000, int(len(mesh.faces) * simplify_faces_ratio))
        )
        report["target_faces"] = int(target_faces)
        if len(reconstructed.faces) > target_faces:
            reconstructed = reconstructed.simplify_quadric_decimation(face_count=target_faces)
            report["decimated_faces"] = int(len(reconstructed.faces))
    elif debug_mode:
        print("[DEBUG] Decimation disabled by ENABLE_DECIMATION flag.")

    if smooth_iterations > 0:
        if debug_mode:
            print(f"[DEBUG] Applying preliminary Laplacian smoothing (iters={smooth_iterations})")
        reconstructed = trimesh.smoothing.filter_laplacian(
            reconstructed, lamb=smooth_lambda, iterations=smooth_iterations
        )

    # 9) Apply isotropic remeshing to enforce the target edge length.
    #    This produces a more uniform mesh suitable for simulation workflows.
    if enable_remesh and target_edge_length is not None:
        if debug_mode:
            print(
                "[DEBUG] Performing True Isotropic Remeshing "
                f"(Edge Length={target_edge_length})..."
            )
        ms = pml.MeshSet()
        ms.add_mesh(pml.Mesh(reconstructed.vertices, reconstructed.faces))
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3,
            targetlen=make_pymeshlab_target_length(target_edge_length),
            adaptive=False,
        )
        out_m = ms.current_mesh()
        reconstructed = trimesh.Trimesh(
            vertices=out_m.vertex_matrix(), faces=out_m.face_matrix(), process=True
        )

        if not reconstructed.is_watertight:
            reconstructed.fill_holes()
            reconstructed.remove_degenerate_faces()

        report["remeshed_vertices"] = len(reconstructed.vertices)

        if post_remesh_smooth_iterations > 0:
            if debug_mode:
                print(
                    "[DEBUG] Applying post-remesh Laplacian relaxation "
                    f"(iters={post_remesh_smooth_iterations})"
                )
            reconstructed = trimesh.smoothing.filter_laplacian(
                reconstructed,
                lamb=smooth_lambda,
                iterations=post_remesh_smooth_iterations,
            )

    # 10) Export mesh and compute quality/size summary metrics.
    #     The report helps compare settings and detect regressions.
    report["post_process"] = "simplify + smooth + remesh + postsmooth"
    reconstructed.export(output_path)
    report["output_path"] = output_path
    report["status"] = "success"

    orig_metric = crinkliness_metric_np(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    rec_metric = crinkliness_metric_np(
        np.asarray(reconstructed.vertices), np.asarray(reconstructed.faces)
    )
    report["crinkliness_orig_mean_deg"] = orig_metric[0]
    report["crinkliness_orig_std_deg"] = orig_metric[1]
    report["crinkliness_rec_mean_deg"] = rec_metric[0]
    report["crinkliness_rec_std_deg"] = rec_metric[1]
    report["input_size_bytes"] = os.path.getsize(input_path)
    report["output_size_bytes"] = os.path.getsize(output_path)

    # 11) Optional visualization of original, marching-cubes, and final meshes.
    #     Why: Quick visual verification of geometry changes across stages.
    if visualize:
        if debug_mode:
            print("\n[DEBUG] Launching multi-panel preview...")
        plotter = pv.Plotter(shape=(1, 3), window_size=[1500, 500])
        pv.set_plot_theme("document")

        plotter.subplot(0, 0)
        plotter.add_text("1. Raw Original Mesh", font_size=10)
        plotter.add_mesh(mesh, color="white", show_edges=True, opacity=0.85, edge_color="gray")

        plotter.subplot(0, 1)
        plotter.add_text("2. Raw TSDF (Marching Cubes)", font_size=10)
        plotter.add_mesh(mc_mesh, color="white", show_edges=True, opacity=0.85, edge_color="gray")

        plotter.subplot(0, 2)
        plotter.add_text("3. Final Smoothed Geometry", font_size=10)
        plotter.add_mesh(reconstructed, color="white", show_edges=True, opacity=0.85, edge_color="gray")

        plotter.link_views()
        png_path = output_path.replace(".obj", "_steps_vis.png")
        if debug_mode:
            print(f"[DEBUG] Saving visualization to {png_path} upon window close.")
        plotter.show(screenshot=png_path)

    return report


def make_pymeshlab_target_length(value: float):
    """Return a pymeshlab length wrapper compatible with installed API version.

    Different pymeshlab versions expose different value wrapper classes.
    This helper picks the first supported class and falls back to a raw float.
    """
    val = float(value)
    if hasattr(pml, "PureValue"):
        return pml.PureValue(val)
    if hasattr(pml, "AbsoluteValue"):
        return pml.AbsoluteValue(val)
    if hasattr(pml, "PercentageValue"):
        return pml.PercentageValue(val)
    return val


def crinkliness_metric_np(vertices: np.ndarray, faces: np.ndarray):
    """Estimate local surface roughness from vertex-normal disagreement.

    Returns mean and standard deviation of angular deviation (degrees) between
    each vertex normal and the average normal of its neighboring vertices.
    """
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    vnorm = np.asarray(tri_mesh.vertex_normals)
    n = vertices.shape[0]
    neighbors = [[] for _ in range(n)]
    for face in faces:
        a, b, c = face
        neighbors[a].extend([b, c])
        neighbors[b].extend([a, c])
        neighbors[c].extend([a, b])
    angles = []
    for i in range(n):
        nbrs = list(set(neighbors[i]))
        if not nbrs:
            continue
        mean_n = np.mean(vnorm[nbrs], axis=0)
        ln = np.linalg.norm(mean_n)
        if ln == 0:
            continue
        mean_n = mean_n / ln
        dot = np.clip(np.dot(vnorm[i], mean_n), -1.0, 1.0)
        angles.append(math.degrees(math.acos(dot)))
    if len(angles) == 0:
        return 0.0, 0.0
    return float(np.mean(angles)), float(np.std(angles))


#%%
def main():
    from pathlib import Path
    script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
    os.chdir(script_dir) # change the working directory
    script_dir = Path(script_dir)

    directory = {}
    directory['home'] = script_dir
    directory['data'] = script_dir / 'patient_atrium_mesh_database'
    directory['result'] = script_dir / 'result'

    # create the result directory if it doesn't exist
    directory['result'].mkdir(exist_ok=True)

    name_prefix = '103_1-lagood'
    input_mesh_path = directory['data'] / f'{name_prefix}.obj'
    output_mesh_path = directory['result'] / f'{name_prefix}_refined.obj'

    # parameter setup
    debug_mode = True
    tsdf_target_res = 120
    tsdf_truncation_dist = None
    morph_closing_iters = 3
    morph_dilation_iters = 1
    pad_voxels = 2
    fill_internal_volume = True
    sdf_smoothing_sigma = 2.0
    mc_level = 0.0
    simplify_faces_ratio = 0.9
    enable_decimation = True
    smooth_iterations = 1
    smooth_lambda = 0.6
    enable_remesh = True
    post_remesh_smooth_iterations = 5
    visualize = False

    report_05mm = clean_mesh(
        str(input_mesh_path),
        str(output_mesh_path),
        debug_mode=debug_mode,
        tsdf_target_res=tsdf_target_res,
        tsdf_truncation_dist=tsdf_truncation_dist,
        morph_closing_iters=morph_closing_iters,
        morph_dilation_iters=morph_dilation_iters,
        pad_voxels=pad_voxels,
        fill_internal_volume=fill_internal_volume,
        sdf_smoothing_sigma=sdf_smoothing_sigma,
        mc_level=mc_level,
        simplify_faces_ratio=simplify_faces_ratio,
        enable_decimation=enable_decimation,
        smooth_iterations=smooth_iterations,
        smooth_lambda=smooth_lambda,
        enable_remesh=enable_remesh,
        target_edge_length=0.5,
        post_remesh_smooth_iterations=post_remesh_smooth_iterations,
        visualize=visualize,
    )
    print("TSDF-style rebuild finished (0.5 mm). Report:")
    for key, value in report_05mm.items():
        print(f"{key}: {value}")
    print()

    debug_plot = 1
    if debug_plot == 1:
        original_mesh = trimesh.load(str(input_mesh_path), force="mesh")
        processed_mesh = trimesh.load(str(output_mesh_path), force="mesh")

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

        for ax, mesh, title in zip(
            axes,
            [original_mesh, processed_mesh],
            ["Original Mesh", "Processed Mesh"],
        ):
            verts = mesh.vertices
            faces = mesh.faces
            poly = Poly3DCollection(
                verts[faces], alpha=0.5, facecolor="white", edgecolor="gray", linewidth=0.1
            )
            ax.add_collection3d(poly)
            ax.set_title(title, fontsize=10)
            ax.view_init(elev=70, azim=-70)
            common.set_axes_equal.execute(ax)

        # plt.tight_layout()
        
        png_path = str(directory['result'] / f'{name_prefix}_mesh_comparison.png')
        plt.savefig(png_path, dpi=300)
        plt.close(fig)

        common.crop_image.execute(png_path)

    print('done')
    return 0

if __name__ == "__main__":
    raise SystemExit(main()) # if this file is run directly, execute main() and exit the program using its return value as the exit code.

#%%