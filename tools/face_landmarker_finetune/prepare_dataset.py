"""Convert 3D OBJ face scans with 16 landmarks to image dataset for training."""

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import trimesh

IMAGE_SIZE = 512


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert OBJ files into images and JSON landmarks")
    parser.add_argument("--input_dir", required=True, help="Directory with OBJ files")
    parser.add_argument("--output_dir", required=True, help="Directory to save images and annotations")
    return parser.parse_args()


def save_mesh_image(mesh: trimesh.Trimesh, out_path: Path) -> np.ndarray:
    """Render the mesh from the front view and save to an image."""
    scene = mesh.scene()
    png = scene.save_image(resolution=(IMAGE_SIZE, IMAGE_SIZE), visible=True)
    out_path.write_bytes(png)
    return png


def extract_landmarks(mesh: trimesh.Trimesh) -> List[np.ndarray]:
    """Assume the first 16 vertices are the landmarks."""
    vertices = np.asarray(mesh.vertices)
    if vertices.shape[0] < 16:
        raise ValueError("Mesh has fewer than 16 vertices")
    return vertices[:16]


def project_landmarks(verts: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """Project landmarks to 2D image coordinates."""
    min_xy = mesh.vertices.min(axis=0)[:2]
    max_xy = mesh.vertices.max(axis=0)[:2]
    coords = verts[:, :2]
    coords_norm = (coords - min_xy) / (max_xy - min_xy)
    coords_pixel = coords_norm * IMAGE_SIZE
    return coords_pixel


def save_landmarks(landmarks: np.ndarray, out_path: Path):
    """Save 16 landmark points to JSON file."""
    data = {
        "landmarks": [
            {"name": str(i + 1), "x": float(x), "y": float(y)}
            for i, (x, y) in enumerate(landmarks)
        ]
    }
    out_path.write_text(json.dumps(data, indent=2))


def process_file(path: Path, output_dir: Path):
    mesh = trimesh.load(path, process=False)
    img_path = output_dir / (path.stem + ".png")
    save_mesh_image(mesh, img_path)
    verts = extract_landmarks(mesh)
    coords = project_landmarks(verts, mesh)
    json_path = output_dir / (path.stem + ".json")
    save_landmarks(coords, json_path)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for obj_path in input_dir.glob("*.obj"):
        process_file(obj_path, output_dir)


if __name__ == "__main__":
    main()
