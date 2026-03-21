#!/usr/bin/env python3
"""
Convert annotations from decimated mesh indices to original mesh indices

Uses nearest-neighbor search to map annotation points on the decimated mesh to the original mesh.
Saves the new annotations as kp_record_new.json
and visualizes the annotation results on the original mesh.

Usage:
    python convert_annotations_to_original_mesh.py /path/to/video
"""

import os
import sys
import json
import numpy as np
from copy import deepcopy

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d not installed")
    sys.exit(1)


def load_and_decimate_mesh(video_dir):
    """Load the original mesh and directly perform decimation"""
    obj_org_path = os.path.join(video_dir, 'obj_org.obj')
    mesh_info_path = os.path.join(video_dir, 'mesh_info.json')

    if not os.path.exists(obj_org_path):
        print(f"Error: obj_org.obj not found")
        return None, None

    try:
        mesh_org = o3d.io.read_triangle_mesh(obj_org_path)
        print(f"OK Original mesh: {len(mesh_org.vertices)} vertices")

        # # Read decimation parameters
        # if os.path.exists(mesh_info_path):
        #     with open(mesh_info_path, 'r', encoding='utf-8') as f:
        #         mesh_info = json.load(f)
        #         target_faces = mesh_info.get('decimation_target_faces', 30000)
        #         vert_threshold = mesh_info.get('decimation_vertex_threshold', 5000)
        #         print(f"  Parameters: target_faces={target_faces}, vert_threshold={vert_threshold}")
        # else:
        #     # Default parameters
        target_faces = 60000
        vert_threshold = 30000
        print(f"  Using default parameters: target_faces={target_faces}, vert_threshold={vert_threshold}")

        # Directly decimate
        mesh_decimated = o3d.geometry.TriangleMesh(mesh_org)
        if len(mesh_org.vertices) > vert_threshold:
            print(f"  Decimating...")
            mesh_decimated = mesh_decimated.simplify_quadric_decimation(
                target_number_of_triangles=target_faces
            )

        print(f"OK Decimated mesh: {len(mesh_decimated.vertices)} vertices")
        return mesh_org, mesh_decimated

    except Exception as e:
        print(f"Error: {e}")
        return None, None


def build_vertex_mapping(mesh_org, mesh_decimated):
    """
    Build mapping from decimated mesh vertices to original mesh vertices.
    For each decimated mesh vertex, find the nearest vertex in the original mesh.
    """
    print("\nBuilding vertex mapping...")

    # Use KDTree to accelerate nearest-neighbor search
    org_tree = o3d.geometry.KDTreeFlann(mesh_org)

    decimated_verts = np.asarray(mesh_decimated.vertices)
    mapping = {}

    for decimated_idx in range(len(decimated_verts)):
        point = decimated_verts[decimated_idx]

        # Search for the nearest vertex in the original mesh
        result = org_tree.search_knn_vector_3d(point, 1)
        org_idx = result[1][0]  # Take the first element of the second return value
        mapping[decimated_idx] = org_idx

    print(f"OK Mapping complete: {len(mapping)} decimated vertices -> original mesh vertices")
    return mapping


def load_kp_record(video_dir):
    """Load original annotations"""
    kp_path = os.path.join(video_dir, 'kp_record_merged.json')
    if not os.path.exists(kp_path):
        print(f"Error: kp_record_merged.json not found")
        return None

    try:
        with open(kp_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading kp_record: {e}")
        return None


def convert_annotations(kp_data, mapping, decimated_vertices_count):
    """
    Convert annotation indices from the decimated mesh to the original mesh.
    """
    print("\nConverting annotation indices...")

    kp_new = {}
    conversion_count = 0
    error_count = 0

    for frame_key, frame_data in kp_data.items():
        # Copy metadata fields
        if frame_key in ['object_scale', 'is_static_object', 'start_frame_index']:
            kp_new[frame_key] = frame_data
            continue

        if not isinstance(frame_data, dict):
            kp_new[frame_key] = frame_data
            continue

        # Convert annotations for this frame
        frame_new = {}

        # Process 2D keypoints
        two_d = frame_data.get('2D_keypoint', [])
        if two_d and isinstance(two_d, list):
            two_d_new = []
            for item in two_d:
                if isinstance(item, list) and len(item) >= 1:
                    decimated_idx = item[0]
                    if isinstance(decimated_idx, int) and decimated_idx in mapping:
                        org_idx = mapping[decimated_idx]
                        # Keep other fields (may contain coordinates, etc.)
                        two_d_new.append([org_idx] + item[1:])
                        conversion_count += 1
                    else:
                        print(f"  Warning: invalid index {decimated_idx} (> {decimated_vertices_count})")
                        error_count += 1
                        two_d_new.append(item)  # Keep original data
                else:
                    two_d_new.append(item)
            if two_d_new:
                frame_new['2D_keypoint'] = two_d_new

        # Process 3D joints
        for key, value in frame_data.items():
            if key == '2D_keypoint':
                continue

            if isinstance(value, int):
                if value in mapping:
                    frame_new[key] = mapping[value]
                    conversion_count += 1
                else:
                    print(f"  Warning: joint {key} index {value} is invalid")
                    error_count += 1
                    frame_new[key] = value  # Keep original data
            else:
                frame_new[key] = value

        kp_new[frame_key] = frame_new

    print(f"OK Conversion complete: {conversion_count} indices converted, {error_count} errors")
    return kp_new


def save_new_annotations(video_dir, kp_new):
    """Save new annotations"""
    kp_new_path = os.path.join(video_dir, 'kp_record_new.json')
    try:
        with open(kp_new_path, 'w', encoding='utf-8') as f:
            json.dump(kp_new, f, indent=2, ensure_ascii=False)
        print(f"\nOK Saved: {kp_new_path}")
        return True
    except Exception as e:
        print(f"Error saving: {e}")
        return False


def extract_annotations_from_kp(kp_data, mesh_vertices_count):
    """Extract all annotated vertex indices from the annotations"""
    valid_indices = set()
    invalid_indices = set()

    for frame_key, frame_data in kp_data.items():
        if frame_key in ['object_scale', 'is_static_object', 'start_frame_index']:
            continue

        if not isinstance(frame_data, dict):
            continue

        # 2D keypoints
        two_d = frame_data.get('2D_keypoint', [])
        if two_d and isinstance(two_d, list):
            for item in two_d:
                if isinstance(item, list) and len(item) >= 1:
                    obj_idx = item[0]
                    if isinstance(obj_idx, int):
                        if obj_idx < mesh_vertices_count:
                            valid_indices.add(obj_idx)
                        else:
                            invalid_indices.add(obj_idx)

        # 3D joints
        for key, value in frame_data.items():
            if key != '2D_keypoint' and isinstance(value, int):
                if value < mesh_vertices_count:
                    valid_indices.add(value)
                else:
                    invalid_indices.add(value)

    return sorted(valid_indices), sorted(invalid_indices)


def visualize_on_original_mesh(video_dir, mesh_org, kp_new):
    """Visualize new annotations on the original mesh"""
    print("\nVisualizing converted annotations...")

    # Extract indices from the new annotations
    valid_indices, invalid_indices = extract_annotations_from_kp(kp_new, len(mesh_org.vertices))

    if not valid_indices:
        print("  No valid annotations")
        return

    # Calculate sphere size
    bbox = mesh_org.get_axis_aligned_bounding_box()
    mesh_size = np.linalg.norm(bbox.max_bound - bbox.min_bound)
    sphere_radius = mesh_size * 0.008

    # Create visualization mesh
    combined_mesh = deepcopy(mesh_org)
    combined_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray

    # Add annotation spheres (red)
    for idx in valid_indices:
        point = mesh_org.vertices[idx]
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(point)
        sphere.paint_uniform_color([1, 0, 0])
        combined_mesh += sphere

    # Add invalid index spheres (orange)
    if invalid_indices:
        for idx in invalid_indices:
            if idx < len(mesh_org.vertices):
                point = mesh_org.vertices[idx]
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                sphere.translate(point)
                sphere.paint_uniform_color([1, 0.5, 0])
                combined_mesh += sphere

    # Save in PLY format (supports vertex colors)
    viz_path_ply = os.path.join(video_dir, 'obj_org_with_annotations_converted.ply')
    o3d.io.write_triangle_mesh(viz_path_ply, combined_mesh)

    # Also save OBJ version (for viewing with other tools)
    viz_path_obj = os.path.join(video_dir, 'obj_org_with_annotations_converted.obj')
    o3d.io.write_triangle_mesh(viz_path_obj, combined_mesh)

    print(f"  OK Annotated vertices: {len(valid_indices)}")
    if invalid_indices:
        print(f"  WARNING Invalid vertices: {len(invalid_indices)} {invalid_indices}")
    print(f"  OK PLY visualization: {os.path.basename(viz_path_ply)} (supports vertex colors)")
    print(f"  OK OBJ visualization: {os.path.basename(viz_path_obj)}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video_dir>")
        print(f"\nExample:")
        print(f"  {sys.argv[0]} /path/to/video")
        sys.exit(1)

    video_dir = sys.argv[1]

    if not os.path.isdir(video_dir):
        print(f"Error: {video_dir} is not a valid directory")
        sys.exit(1)

    print("\n" + "="*70)
    print("Annotation index conversion: decimated mesh -> original mesh")
    print("="*70)
    print(f"Video directory: {video_dir}\n")

    # 1. Load mesh and decimate
    mesh_org, mesh_decimated = load_and_decimate_mesh(video_dir)
    if mesh_org is None or mesh_decimated is None:
        return 1

    # 2. Build vertex mapping
    mapping = build_vertex_mapping(mesh_org, mesh_decimated)

    # 3. Load annotations
    kp_data = load_kp_record(video_dir)
    if kp_data is None:
        return 1

    # 4. Convert annotations
    kp_new = convert_annotations(kp_data, mapping, len(mesh_decimated.vertices))

    # 5. Save new annotations
    if not save_new_annotations(video_dir, kp_new):
        return 1

    # 6. Visualize
    visualize_on_original_mesh(video_dir, mesh_org, kp_new)

    print("\n" + "="*70)
    print("DONE Conversion complete!")
    print("="*70)
    print(f"New annotation file: kp_record_new.json")
    print(f"Visualization file: obj_org_with_annotations_converted.obj")

    return 0


if __name__ == '__main__':
    sys.exit(main())
