# Copyright 2026 Jiyue He
# Licensed under the Apache License, Version 2.0

"""Flask/Three.js viewer for all-voxel cardiac activation movies."""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, render_template, send_from_directory


TOOL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TOOL_DIR.parent
RESULT_SUFFIX = "_simulation_results_2998.npz"
MESH_SUFFIX = "_mesh.npz"


def _latest_match(folders: list[Path], pattern: str) -> Path | None:
    matches: list[Path] = []
    for folder in folders:
        if folder.is_dir():
            matches.extend(folder.glob(pattern))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _default_result_path() -> Path:
    environment_path = os.environ.get("PYHEARTSIM_RESULT")
    if environment_path:
        return Path(environment_path).expanduser()
    result_path = _latest_match(
        [PROJECT_DIR / "result", Path("/home/j/Desktop/hdd/share_folder/simulation_results")],
        f"*{RESULT_SUFFIX}",
    )
    if result_path is None:
        raise FileNotFoundError(
            "No simulation result was found. Pass --result or set PYHEARTSIM_RESULT."
        )
    return result_path


def _default_mesh_path(result_path: Path) -> Path:
    environment_path = os.environ.get("PYHEARTSIM_MESH")
    if environment_path:
        return Path(environment_path).expanduser()
    if not result_path.name.endswith(RESULT_SUFFIX):
        raise ValueError(f"Cannot infer a mesh from {result_path.name!r}; pass --mesh.")

    prefix = result_path.name[: -len(RESULT_SUFFIX)]
    mesh_name = f"{prefix}{MESH_SUFFIX}"
    candidates = [
        result_path.parent / mesh_name,
        PROJECT_DIR / "data" / mesh_name,
        Path("/home/j/Desktop/hdd/share_folder/carto3_files/data npz") / mesh_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No matching mesh was found. Pass --mesh or set PYHEARTSIM_MESH. "
        f"Searched:\n  {searched}"
    )


def _float32(values: np.ndarray) -> np.ndarray:
    """Return C-contiguous little-endian float32 data for browser transfer."""
    return np.ascontiguousarray(values, dtype="<f4")


def load_movie_data(
    mesh_path: Path, result_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file does not exist: {mesh_path}")
    if not result_path.is_file():
        raise FileNotFoundError(f"Simulation result does not exist: {result_path}")

    with np.load(mesh_path, allow_pickle=False) as archive:
        if "voxel3mm_1mm_spacing" not in archive.files:
            raise KeyError(f"{mesh_path} does not contain 'voxel3mm_1mm_spacing'")
        voxel = _float32(archive["voxel3mm_1mm_spacing"])

    # NPZ members load lazily, so h and electrogram_unipolar remain unloaded.
    with np.load(result_path, allow_pickle=False) as archive:
        missing = {"action_potential_electrode", "physical_time"}.difference(archive.files)
        if missing:
            raise KeyError(
                f"Missing {', '.join(sorted(missing))}. Run the simulation with "
                "save_action_potential_of_all_voxel_flag = 1."
            )
        action_potential = _float32(archive["action_potential_electrode"])
        physical_time = _float32(archive["physical_time"])

    if voxel.ndim != 2 or voxel.shape[1] != 3:
        raise ValueError(f"voxel must have shape (n_voxel, 3), got {voxel.shape}")
    if action_potential.ndim != 2:
        raise ValueError(f"action_potential must be 2D, got {action_potential.shape}")
    if physical_time.ndim != 1:
        raise ValueError(f"physical_time must be 1D, got {physical_time.shape}")
    if action_potential.shape[1] != voxel.shape[0]:
        raise ValueError(
            f"Mesh has {voxel.shape[0]} voxels but each movie frame has "
            f"{action_potential.shape[1]} values."
        )
    if action_potential.shape[0] != physical_time.shape[0]:
        raise ValueError(
            f"Movie has {action_potential.shape[0]} frames but physical_time has "
            f"{physical_time.shape[0]} samples."
        )
    if action_potential.shape[0] == 0:
        raise ValueError("The simulation result contains no movie frames.")
    return voxel, action_potential, physical_time


def create_app(mesh_path: Path, result_path: Path) -> Flask:
    mesh_path = mesh_path.resolve()
    result_path = result_path.resolve()
    voxel, action_potential, physical_time = load_movie_data(mesh_path, result_path)

    app = Flask(__name__, template_folder=str(TOOL_DIR), static_folder=None)
    app.config.update(
        MESH_PATH=str(mesh_path),
        RESULT_PATH=str(result_path),
        VOXEL_COUNT=int(voxel.shape[0]),
        FRAME_COUNT=int(action_potential.shape[0]),
    )

    @app.get("/")
    def index():
        return render_template("ui_movie.html")

    @app.get("/ui_movie.css")
    def serve_css():
        return send_from_directory(TOOL_DIR, "ui_movie.css")

    @app.get("/api/metadata")
    def metadata():
        name = (
            result_path.name[: -len(RESULT_SUFFIX)]
            if result_path.name.endswith(RESULT_SUFFIX)
            else result_path.stem
        )
        return jsonify(
            name=name,
            voxel_count=int(voxel.shape[0]),
            frame_count=int(action_potential.shape[0]),
            physical_time=physical_time.tolist(),
            data_min=0.0,
            data_max=1.0,
            activation_threshold=0.13,
        )

    @app.get("/api/voxels")
    def voxels():
        response = Response(voxel.tobytes(), mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "public, max-age=3600"
        response.headers["X-Array-Dtype"] = "float32-little-endian"
        response.headers["X-Array-Shape"] = f"{voxel.shape[0]},3"
        return response

    @app.get("/api/frame/<int:frame_id>")
    def frame(frame_id: int):
        if frame_id >= action_potential.shape[0]:
            return jsonify(error=f"Frame {frame_id} is out of range."), 404
        values = action_potential[frame_id]
        response = Response(values.tobytes(), mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "public, max-age=3600"
        response.headers["X-Array-Dtype"] = "float32-little-endian"
        response.headers["X-Array-Shape"] = str(values.shape[0])
        return response

    @app.get("/api/health")
    def health():
        return jsonify(
            status="ok",
            voxel_count=int(voxel.shape[0]),
            frame_count=int(action_potential.shape[0]),
        )

    return app


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a PyHeartSim all-voxel result in a web browser."
    )
    parser.add_argument("--result", type=Path, help="Path to *_simulation_results.npz")
    parser.add_argument("--mesh", type=Path, help="Path to the matching *_mesh.npz")
    parser.add_argument("--host", default="127.0.0.1", help="Flask bind host")
    parser.add_argument("--port", default=5002, type=int, help="Flask bind port")
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open the browser automatically"
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    result_path = (
        arguments.result.expanduser() if arguments.result else _default_result_path()
    )
    mesh_path = (
        arguments.mesh.expanduser()
        if arguments.mesh
        else _default_mesh_path(result_path)
    )
    app = create_app(mesh_path, result_path)
    url_host = "127.0.0.1" if arguments.host in {"0.0.0.0", "::"} else arguments.host
    url = f"http://{url_host}:{arguments.port}"

    print(f"Mesh:  {mesh_path.resolve()}")
    print(f"Result: {result_path.resolve()}")
    print(
        f"Movie:  {app.config['FRAME_COUNT']} frames on "
        f"{app.config['VOXEL_COUNT']} voxels"
    )
    print(f"Viewer: {url}")
    if not arguments.no_browser:
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    app.run(
        debug=False,
        host=arguments.host,
        port=arguments.port,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
