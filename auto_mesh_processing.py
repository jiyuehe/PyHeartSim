#%%
import argparse
import inspect
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pymeshlab as pml # pip install pymeshlab
import pyvista as pv # pip install pyvista
import trimesh # pip install trimesh
from scipy import ndimage
from skimage import measure # pip install scikit-image


@dataclass
class TSDFConfig:
    debug_mode: bool = True
    tsdf_target_res: int = 120
    tsdf_truncation_dist: Optional[float] = None
    morph_closing_iters: int = 3
    morph_dilation_iters: int = 1
    pad_voxels: int = 2
    fill_internal_volume: bool = True
    sdf_smoothing_sigma: float = 2.0
    mc_level: float = 0.0
    simplify_faces_ratio: float = 0.9
    enable_decimation: bool = True
    smooth_iterations: int = 10
    smooth_lambda: float = 0.6
    enable_remesh: bool = True
    target_edge_length: Optional[float] = 5.0
    post_remesh_smooth_iterations: int = 5
    visualize: bool = False


def decimate_mesh_with_best_api(tri_mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    decimator = None
    if hasattr(tri_mesh, "simplify_quadric_decimation"):
        decimator = tri_mesh.simplify_quadric_decimation
    elif hasattr(tri_mesh, "simplify_quadratic_decimation"):
        decimator = tri_mesh.simplify_quadratic_decimation

    if decimator is None:
        raise RuntimeError("No quadric decimator is available on this trimesh build.")

    current_faces = max(int(len(tri_mesh.faces)), 1)
    target_faces = int(max(4, target_faces))
    keep_ratio = float(np.clip(target_faces / float(current_faces), 0.001, 1.0))
    target_reduction = float(np.clip(1.0 - keep_ratio, 0.0, 0.999))

    try:
        params = set(inspect.signature(decimator).parameters.keys())
    except Exception:
        params = set()

    attempts = [
        ("face_count", target_faces),
        ("target_count", target_faces),
        ("target_faces", target_faces),
        ("target_reduction", target_reduction),
        ("percent", keep_ratio),
    ]

    last_error = None
    for param, value in attempts:
        if params and param not in params:
            continue
        try:
            return decimator(**{param: value})
        except Exception as err:
            last_error = err

    for value in (target_faces, target_reduction):
        try:
            return decimator(value)
        except Exception as err:
            last_error = err

    raise RuntimeError(f"Decimation failed across available call signatures: {last_error}")


def make_pymeshlab_target_length(value: float):
    val = float(value)
    if hasattr(pml, "PureValue"):
        return pml.PureValue(val)
    if hasattr(pml, "AbsoluteValue"):
        return pml.AbsoluteValue(val)
    if hasattr(pml, "PercentageValue"):
        return pml.PercentageValue(val)
    return val


def laplacian_smooth(tri_mesh: trimesh.Trimesh, iterations: int = 3, lamb: float = 0.5) -> trimesh.Trimesh:
    v = tri_mesh.vertices.copy()
    f = tri_mesh.faces
    n = v.shape[0]
    neighbors = [[] for _ in range(n)]
    for face in f:
        a, b, c = face
        neighbors[a].extend([b, c])
        neighbors[b].extend([a, c])
        neighbors[c].extend([a, b])
    for i in range(n):
        neighbors[i] = list(set(neighbors[i]))

    for _ in range(iterations):
        v_next = v.copy()
        for i in range(n):
            nbrs = neighbors[i]
            if not nbrs:
                continue
            mean_n = np.mean(v[nbrs], axis=0)
            v_next[i] = v[i] + lamb * (mean_n - v[i])
        v = v_next
    tri_mesh.vertices = v
    return tri_mesh


def compute_vertex_normals_np(vertices: np.ndarray, faces: np.ndarray):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    fn_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
    fn_len[fn_len == 0] = 1.0
    face_normals = face_normals / fn_len
    vnorm = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vnorm, faces[:, i], face_normals)
    vn_len = np.linalg.norm(vnorm, axis=1, keepdims=True)
    vn_len[vn_len == 0] = 1.0
    vnorm = vnorm / vn_len
    return vnorm, face_normals


def crinkliness_metric_np(vertices: np.ndarray, faces: np.ndarray):
    vnorm, _ = compute_vertex_normals_np(vertices, faces)
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


class TSDFRebuilder:
    def __init__(self, config: Optional[TSDFConfig] = None):
        self.config = config or TSDFConfig()

    def run(self, input_path: str, output_path: str) -> dict:
        cfg = self.config
        report = {"status": "starting", "input_used": input_path, "config": asdict(cfg)}

        mesh = trimesh.load(input_path, force="mesh")
        if mesh.is_empty:
            raise RuntimeError("Loaded mesh is empty.")

        if not mesh.is_watertight:
            if cfg.debug_mode:
                print("[DEBUG] Input mesh is not watertight, attempting repairs...")
            try:
                mesh.remove_duplicate_faces()
                mesh.remove_unreferenced_vertices()
                mesh.fill_holes()
            except Exception as err:
                if cfg.debug_mode:
                    print(f"[DEBUG] Repair attempt failed: {err}")

        bbox = mesh.bounds
        extent = bbox[1] - bbox[0]
        diag = np.linalg.norm(extent)
        pitch = (
            float(max(extent) / cfg.tsdf_target_res)
            if max(extent) > 0
            else (diag / cfg.tsdf_target_res if diag > 0 else 0.001)
        )
        report["grid_pitch"] = float(pitch)
        if cfg.debug_mode:
            print(f"[DEBUG] Grid pitch set to: {pitch}")

        voxelized = mesh.voxelized(pitch)
        vox_matrix = voxelized.matrix.copy()
        report["vox_shape"] = vox_matrix.shape
        report["vox_filled_count"] = int(np.count_nonzero(vox_matrix))
        if cfg.debug_mode:
            print(
                f"[DEBUG] Initial voxelization complete. Voxel shape {vox_matrix.shape}, "
                f"filled voxels: {report['vox_filled_count']}"
            )

        occ = vox_matrix.astype(bool)
        if cfg.pad_voxels > 0:
            occ = np.pad(occ, pad_width=cfg.pad_voxels, mode="constant", constant_values=False)
            if cfg.debug_mode:
                print(f"[DEBUG] Grid padded by {cfg.pad_voxels} voxels to prevent mesh border holes.")

        if cfg.morph_dilation_iters > 0:
            occ = ndimage.binary_dilation(occ, iterations=cfg.morph_dilation_iters)
            if cfg.debug_mode:
                print(
                    f"[DEBUG] Morphological dilation applied (iters={cfg.morph_dilation_iters}). "
                    f"New filled: {np.count_nonzero(occ)}"
                )

        if cfg.morph_closing_iters > 0:
            occ = ndimage.binary_closing(occ, iterations=cfg.morph_closing_iters)
            if cfg.debug_mode:
                print(
                    f"[DEBUG] Morphological closing applied (iters={cfg.morph_closing_iters}). "
                    f"New filled: {np.count_nonzero(occ)}"
                )

        # if cfg.debug_mode:
        #     points_grid = np.argwhere(occ) * pitch + (bbox[0] - (pitch / 2.0))
        #     trimesh.points.PointCloud(points_grid).export("debug_voxel_cloud.ply")
        #     print("[DEBUG] Exported intermediate occupied voxel centers to 'debug_voxel_cloud.ply'")

        if cfg.fill_internal_volume:
            occ = ndimage.binary_fill_holes(occ)
            if cfg.debug_mode:
                print(
                    "[DEBUG] Binary Hole Filling applied to discard internal walls. "
                    f"New filled voxels: {np.count_nonzero(occ)}"
                )

        outside_dist = ndimage.distance_transform_edt(~occ) * pitch
        inside_dist = ndimage.distance_transform_edt(occ) * pitch
        sdf = outside_dist.copy()
        sdf[occ] = -inside_dist[occ]

        if cfg.tsdf_truncation_dist is not None:
            sdf = np.clip(sdf, -cfg.tsdf_truncation_dist, cfg.tsdf_truncation_dist)

        if cfg.sdf_smoothing_sigma > 0:
            sdf = ndimage.gaussian_filter(sdf, sigma=cfg.sdf_smoothing_sigma)
            if cfg.debug_mode:
                print(f"[DEBUG] Gaussian smoothing applied on SDF with sigma={cfg.sdf_smoothing_sigma}.")

        origin = bbox[0] - (pitch / 2.0) - (pitch * cfg.pad_voxels)
        report["assumed_origin"] = origin.tolist()

        if cfg.debug_mode:
            print(f"[DEBUG] Running Marching Cubes at iso-level {cfg.mc_level}...")

        try:
            verts, faces, _, _ = measure.marching_cubes(
                sdf, level=cfg.mc_level, spacing=(pitch, pitch, pitch)
            )
        except Exception:
            verts, faces, _, _ = measure.marching_cubes_lewiner(
                sdf, level=cfg.mc_level, spacing=(pitch, pitch, pitch)
            )

        verts_world = verts + origin
        reconstructed = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
        mc_mesh = reconstructed.copy()

        report["reconstructed_vertices"] = int(len(reconstructed.vertices))
        report["reconstructed_faces"] = int(len(reconstructed.faces))

        if cfg.enable_decimation:
            target_faces = min(
                len(reconstructed.faces), max(10000, int(len(mesh.faces) * cfg.simplify_faces_ratio))
            )
            report["target_faces"] = int(target_faces)
            if len(reconstructed.faces) > target_faces:
                try:
                    reconstructed = decimate_mesh_with_best_api(reconstructed, target_faces)
                    report["decimated_faces"] = int(len(reconstructed.faces))
                except Exception as dec_err:
                    if cfg.debug_mode:
                        print(f"[DEBUG] Decimation skipped: {dec_err}")
                    report["decimation_skipped_reason"] = str(dec_err)
        elif cfg.debug_mode:
            print("[DEBUG] Decimation disabled by ENABLE_DECIMATION flag.")

        if cfg.smooth_iterations > 0:
            if cfg.debug_mode:
                print(f"[DEBUG] Applying preliminary Laplacian smoothing (iters={cfg.smooth_iterations})")
            reconstructed = laplacian_smooth(
                reconstructed, iterations=cfg.smooth_iterations, lamb=cfg.smooth_lambda
            )

        if cfg.enable_remesh and cfg.target_edge_length is not None:
            if cfg.debug_mode:
                print(
                    "[DEBUG] Performing True Isotropic Remeshing "
                    f"(Edge Length={cfg.target_edge_length})..."
                )
            ms = pml.MeshSet()
            ms.add_mesh(pml.Mesh(reconstructed.vertices, reconstructed.faces))
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3,
                targetlen=make_pymeshlab_target_length(cfg.target_edge_length),
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

            if cfg.post_remesh_smooth_iterations > 0:
                if cfg.debug_mode:
                    print(
                        "[DEBUG] Applying post-remesh Laplacian relaxation "
                        f"(iters={cfg.post_remesh_smooth_iterations})"
                    )
                reconstructed = laplacian_smooth(
                    reconstructed,
                    iterations=cfg.post_remesh_smooth_iterations,
                    lamb=cfg.smooth_lambda,
                )

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

        self._maybe_visualize(mesh, mc_mesh, reconstructed, output_path)
        return report

    def _maybe_visualize(
        self,
        mesh: trimesh.Trimesh,
        mc_mesh: trimesh.Trimesh,
        reconstructed: trimesh.Trimesh,
        output_path: str,
    ) -> None:
        if not self.config.visualize:
            return

        if self.config.debug_mode:
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
        if self.config.debug_mode:
            print(f"[DEBUG] Saving visualization to {png_path} upon window close.")
        plotter.show(screenshot=png_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TSDF-style mesh rebuild pipeline.")
    parser.add_argument("--input", default="patient_atrium_mesh_database/105_6-LA.obj", help="Input mesh path")
    parser.add_argument("--output", default="./result/105_6-LA_refined.obj", help="Output mesh path")
    parser.add_argument("--visualize", action="store_true", help="Show 3-panel visualization")
    parser.add_argument("--debug-mode", action="store_true", default=True, help="Enable debug logs")
    parser.add_argument("--no-debug-mode", action="store_false", dest="debug_mode")
    parser.add_argument("--target-res", type=int, default=120, help="Voxel grid target resolution")
    parser.add_argument("--truncation-dist", type=float, default=None, help="TSDF truncation distance")
    parser.add_argument("--morph-closing-iters", type=int, default=3)
    parser.add_argument("--morph-dilation-iters", type=int, default=1)
    parser.add_argument("--pad-voxels", type=int, default=2)
    parser.add_argument("--fill-internal-volume", action="store_true", default=True)
    parser.add_argument("--no-fill-internal-volume", action="store_false", dest="fill_internal_volume")
    parser.add_argument("--sdf-smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--mc-level", type=float, default=0.0)
    parser.add_argument("--simplify-faces-ratio", type=float, default=0.9)
    parser.add_argument("--enable-decimation", action="store_true", default=True)
    parser.add_argument("--disable-decimation", action="store_false", dest="enable_decimation")
    parser.add_argument("--smooth-iterations", type=int, default=10)
    parser.add_argument("--smooth-lambda", type=float, default=0.6)
    parser.add_argument("--enable-remesh", action="store_true", default=True)
    parser.add_argument("--disable-remesh", action="store_false", dest="enable_remesh")
    parser.add_argument("--target-edge-length", type=float, default=5.0)
    parser.add_argument("--post-remesh-smooth-iterations", type=int, default=5)
    return parser


def config_from_args(args: argparse.Namespace) -> TSDFConfig:
    return TSDFConfig(
        debug_mode=args.debug_mode,
        tsdf_target_res=args.target_res,
        tsdf_truncation_dist=args.truncation_dist,
        morph_closing_iters=args.morph_closing_iters,
        morph_dilation_iters=args.morph_dilation_iters,
        pad_voxels=args.pad_voxels,
        fill_internal_volume=args.fill_internal_volume,
        sdf_smoothing_sigma=args.sdf_smoothing_sigma,
        mc_level=args.mc_level,
        simplify_faces_ratio=args.simplify_faces_ratio,
        enable_decimation=args.enable_decimation,
        smooth_iterations=args.smooth_iterations,
        smooth_lambda=args.smooth_lambda,
        enable_remesh=args.enable_remesh,
        target_edge_length=args.target_edge_length,
        post_remesh_smooth_iterations=args.post_remesh_smooth_iterations,
        visualize=args.visualize,
    )

#%%
def main() -> int:
    import os
    from pathlib import Path
    script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
    os.chdir(script_dir) # change the working directory
    script_dir = Path(script_dir)

    directory = {}
    directory['home'] = script_dir
    directory['data'] = script_dir / 'patient_atrium_mesh_database'
    directory['result'] = script_dir / 'result'

    name_prefix = '105_6-LA'
    input_mesh_path = directory['data'] / f'{name_prefix}.obj'
    output_mesh_path = directory['result'] / f'{name_prefix}_refined.obj'
    output_mesh_path_3mm = directory['result'] / f'{name_prefix}_refined_3mm.obj'

    args = build_parser().parse_args()
    cfg = config_from_args(args)
    report = TSDFRebuilder(cfg).run(args.input, args.output)
    print("TSDF-style rebuild finished. Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # mesh folder is directory['data']
    # output folder is directory['result']
    # make all the parameter settings right here, explicitly: delete build_parser() and config_from_args(), and put their contents here directly
    # then do the processing
    # then save the processed mesh (edge length of 0.5 mm) to folder directory['result'] and save it as 105_6-LA_refined.obj
    # then save another mesh (edge length of 3 mm) to folder directory['result'] and save it as 105_6-LA_refined_3mm.obj

    # clean up the code a bit: 
    # remove all the unnecessary protection codes: those 'try-except', and the many 'if', some of them looks like not necessary
    # we will not use command line arguments, so remove all the argparse related code, and just set the parameters directly in the code

    print('done')
    return 0

if __name__ == "__main__":
    raise SystemExit(main()) # if this file is run directly, execute main() and exit the program using its return value as the exit code.

#%%