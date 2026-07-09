# Coordinates

All GeoSS geometry code uses explicit matrix names:

- `c2w`: camera-to-world.
- `w2c`: world-to-camera.
- OpenCV camera axes: x right, y down, z forward.
- GeoSS canonical anchor cube: `[-1,1]^3`.
- TRELLIS sparse-structure voxel unit cube is treated as `[0,1]^3` when converting occupied coordinates.
- MeshFleet_TRELLIS voxel PLY files in the inspected local sample are centered
  around zero with max abs <= 0.5. They are treated as TRELLIS centered
  half-cube coordinates and converted to GeoSS canonical by `xyz * 2`.

Objaverse/Blender/OpenGL cameras are converted with:

```text
diag([1, -1, -1, 1])
```

Projection utilities:

- `project_points(points_world, K, w2c)`
- `unproject_depth(depth, K, c2w)`
- `generate_camera_rays(H, W, K, c2w)`
- `projection_roundtrip_check(points_world, K, c2w)`

Mapping utilities:

- `canonical_to_trellis_grid(points, resolution)`: `[-1,1]^3 -> [0,R)` continuous grid.
- `trellis_grid_to_canonical(grid_xyz, resolution)`: `[0,R) -> [-1,1]^3`.
- `anchor_to_occ_index(anchor_xyz, resolution)`: canonical anchors to nearest occupancy index.
- `occ_index_to_anchor_center(indices, resolution)`: occupancy indices to canonical voxel centers.

MeshFleet_TRELLIS mapping:

```text
voxel_ply xyz in [-0.5,0.5]^3
-> GeoSS canonical xyz = clamp(xyz * 2, -1, 1)
-> occupancy index = floor(((canonical + 1) / 2) * R)
```

The round-trip test uses voxel centers, not voxel corners:

```text
index -> index + 0.5 -> canonical center -> index
```

Main risks:

- VGGT returns OpenCV `w2c`; Objaverse typically stores OpenGL `c2w`.
- TRELLIS uses dense latent grids for SS flow, while GeoSS uses sparse anchors and tokenized velocity control.
- VGGT feature tensor formats can change across versions, so the wrapper keeps raw tokens and normalized geometry outputs separate.
