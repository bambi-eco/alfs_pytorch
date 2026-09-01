> This project is archived: We migrated the pytorch implementation to the main alfs_py implementation (https://github.com/bambi-eco/alfs_py/)

# Python based Airborne Light Field Sampling and Orthographic Projection

A Python-based framework for Airborne Light-Field Sampling (ALFS) data visualization and orthographic projection of geo-referenced drone footage, implemented via **PyTorch**.

> **This is the PyTorch port of ALFSPy.** The rendering backend no longer uses OpenGL:
> there is no ModernGL, no GL driver, no X server and no Xvfb. Rendering runs on a
> tensor-based rasteriser (`alfspy.core.torchgl`) on CPU or CUDA, which removes the
> intermittent buffer artifacts the containerised ModernGL deployment suffered from.
>
> The public API is unchanged — `Renderer`, `CtxShot`, `ProjectionScene` and the pipeline
> scripts keep their signatures. See [`docs/MIGRATION_REVIEW.md`](docs/MIGRATION_REVIEW.md)
> for what changed and why, and [`test/README.md`](test/README.md) for the test suite.
>
> Equivalence with the ModernGL renderer is verified by an opt-in parity gate: mean absolute
> error below one 8-bit level on every scene tested, with disagreements confined to coverage
> boundaries. SharePoint support has been removed.

## Quick start

```bash
# Install torch for your platform first; the CPU build is much smaller.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .

pytest -q          # 150 tests, no GPU or GL driver required
```

Selecting a device:

```python
from alfspy.render.projection import ProjectionScene, ProjectionSettings

with ProjectionScene(..., device="cuda") as scene:    # or "cpu"; defaults to CUDA if present
    scene.project_orthographic(...)
```

## Overview

This framework provides tools for processing and rendering (drone-based thermal and RGB) imagery with two primary rendering modes:

1. **Orthographic Projection**: Generates top-down, orthorectified views of captured footage by projecting image data onto a Digital Elevation Model (DEM)
2. **Airborne Light Field Sampling (ALFS)**: Synthesizes novel views by integrating multiple overlapping drone captures, enabling reconstruction-free radiance field rendering


---

## Theoretical Background

### Airborne Light Field Sampling (ALFS)

Traditional approaches to generating novel aerial views typically require extensive 3D reconstruction through photogrammetry or neural radiance fields. ALFS offers a reconstruction-free alternative that directly synthesizes views from captured drone imagery.

#### Core Concept

The ALFS approach operates on the principle that drone flight paths can be designed to capture overlapping views of a scene from multiple positions. When these captures are combined using projective geometry, novel viewpoints can be synthesized without explicit 3D reconstruction.

The key insight is that by:
1. Projecting each captured image onto a known terrain surface (DEM)
2. Integrating the contributions from multiple overlapping shots
3. Rendering from a virtual camera position

...we can generate high-quality novel views that preserve the visual characteristics of the original captures.

#### Mathematical Foundation

For each pixel in the output image, the rendering process:

1. **Casts a ray** from the virtual camera through the pixel into the scene
2. **Intersects** the ray with the Digital Elevation Model to find the world-space point
3. **Projects** this world-space point back into each source image using the inverse camera matrices
4. **Samples** the color from each source image where valid
5. **Integrates** the samples using additive blending with normalization


### Orthographic Projection

Orthographic projection provides a distortion-free, top-down view of the captured scene. Unlike perspective projection, parallel lines remain parallel in orthographic projection, making it ideal for:

- Geographic mapping and analysis
- Consistent size measurements across the image
- Wildlife detection dataset preparation
- Integration with GIS systems

The orthographic camera is defined by:
- **Position**: Center point above the scene
- **Orthographic Size**: Width and height in world units (meters)
- **Resolution**: Output image dimensions in pixels

---

## Architecture

### Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Flight Data    │────▶│  Pre-Processing │────▶│    Rendering    │
│  (Images + GPS) │     │  (Calibration)  │     │  (ALFS/Ortho)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   ┌─────────┐           ┌───────────┐          ┌──────────────┐
   │ Thermal │           │   DEM     │          │ Output Image │
   │  RGB    │           │   Poses   │          │    Labels    │
   │ Metadata│           │   Mask    │          └──────────────┘
   └─────────┘           └───────────┘
```

### Core Components

#### 1. Camera System (`src/alfspy/core/rendering/camera.py`)

The camera system supports both perspective and orthographic projections:

```python
class Camera:
    def __init__(self, 
                 fovy: float = 60.0,           # Field of view (perspective)
                 aspect_ratio: float = 1.0,     # Width/height ratio
                 orthogonal: bool = False,      # Enable orthographic mode
                 orthogonal_size: tuple = (16, 16),  # World-space dimensions
                 near: float = 0.1,             # Near clipping plane
                 far: float = 10000,            # Far clipping plane
                 position: Vector3 = ...,       # Camera position
                 rotation: Quaternion = ...)    # Camera orientation
```

#### 2. Shot Representation (`src/alfspy/core/rendering/shot.py`)

Each captured image is represented as a `CtxShot` containing:
- **Camera parameters**: Position, rotation, field of view
- **Image data**: Texture for GPU rendering
- **Correction transform**: Compensates for GPS/IMU drift

```python
class CtxShot:
    camera: Camera              # Source camera parameters
    correction: Transform       # Position/rotation correction
    tex: Texture               # GPU texture handle
```

#### 3. Renderer (`src/alfspy/core/rendering/renderer.py`)

The GPU-accelerated renderer handles both single-shot projection and multi-shot integration:

**Single Shot Projection**:
```python
renderer.project_shots(shots, RenderResultMode.ShotOnly, mask=mask)
```

**Light Field Integration**:
```python
renderer.render_integral(shots, mask=mask, alpha_threshold=0.1)
```

Integration accumulates additively into a tensor. The former fragment shader is now
`alfspy.core.torchgl.programs.shade_shot`, which quotes the GLSL it replaces:

```python
uv = clip[:, :3] / w.unsqueeze(-1) / 2.0 + 0.5
keep = (uv[:, 0] >= 0) & (uv[:, 0] <= 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= 1)
colour = texture.sample(uv[keep, 0], uv[keep, 1])
if mask is not None:
    colour = colour * mask.sample(uv[keep, 0], uv[keep, 1])[:, 0:1]
```

The alpha channel doubles as an **overlap counter**: each contributing shot adds `1.0`, so
dividing by alpha normalises each pixel by how many shots actually saw it.

#### 4. Coordinate Conversion (`src/alfspy/core/convert/convert.py`)

Bidirectional conversion between pixel and world coordinates:

**Pixel → World**:
```python
world_coords = pixel_to_world_coord(
    x, y,                    # Pixel coordinates
    width, height,           # Image dimensions
    mesh,                    # DEM as trimesh
    camera,                  # Source camera
    distortion=None,         # Lens distortion model
    include_misses=True      # Include failed projections
)
```

**World → Pixel**:
```python
pixel_coords = world_to_pixel_coord(
    coordinates,             # 3D world points
    width, height,           # Target image dimensions
    camera                   # Target camera
)
```

---

## Rendering Modes

### Orthographic Projection Mode

In orthographic mode (`project_orthogonal=True`), each frame is independently projected onto the DEM and rendered from directly above:

```python
settings = BaseSettings(
    orthogonal=True,
    ortho_size=(ORTHO_WIDTH, ORTHO_HEIGHT),  # World units (meters)
    camera_dist=CAMERA_DISTANCE,              # Height above terrain
    resolution=Resolution(RENDER_WIDTH, RENDER_HEIGHT)
)
```

**Use Cases**:
- Creating training datasets for object detection
- Geographic mapping of animal positions
- Consistent scale imagery for size estimation

### ALFS Integration Mode

In ALFS mode (`project_orthogonal=False`), multiple frames are combined:

```python
# Collect neighboring frames
prev_shots = get_shots_for_files(previous_frames, ...)
add_shots = get_shots_for_files(additional_frames, ...)
all_shots = prev_shots + [current_shot] + add_shots

# Render integrated view
renderer.render_integral(
    shot_loader,
    mask=mask,
    save=True,
    alpha_threshold=alpha_threshold
)
```

**Key Parameters**:
- `nr_of_frames_before_current`: Temporal window (past)
- `nr_of_frames_after_current`: Temporal window (future)
- `neighbor_fps`: Sampling rate within temporal window
- `alpha_threshold`: Minimum overlap count for valid pixels
- `merge_labels_in_alfs`: Aggregate labels from all contributing frames

**Use Cases**:
- Noise reduction through multi-view integration
- Novel viewpoint synthesis
- Gap filling for occluded regions

## Label Projection

The framework includes tools for projecting 2D bounding box labels from source images to orthographic outputs:

### Workflow

1. **Read source labels** in YOLO format (class, x_center, y_center, width, height)
2. **Convert to pixel coordinates** in source image space
3. **Project to world coordinates** via ray-mesh intersection
4. **Re-project to output image** using the virtual camera
5. **Compute axis-aligned bounding boxes** in output space
6. **Export in YOLO format** for training

```python
# Project label coordinates
w_poses = pixel_to_world_coord(pixel_xs, pixel_ys, 
                                input_resolution.width, input_resolution.height,
                                tri_mesh, camera, include_misses=False)

np_poses = world_to_pixel_coord(w_poses, 
                                 render_resolution.width, render_resolution.height,
                                 single_shot_camera)

# Convert to axis-aligned bounding box
axis_aligned_bb = get_axis_aligned_bounding_box(projected_coords)
yolo_label = to_yolo_format(axis_aligned_bb, img_width, img_height)
```

### Label Merging in ALFS Mode

When `merge_labels_in_alfs=True`, labels from all contributing frames are aggregated:
- Labels are collected from each frame in the temporal window
- Non-maximum suppression (NMS) can be applied to remove duplicates
- The `NMS_IOU` parameter controls the overlap threshold

---

## Correction System

GPS and IMU measurements from drones sometimes contain systematic errors that must be corrected for accurate geo-referencing.

### Correction Transform

Each shot can have position and rotation corrections:

```python
correction = Transform(
    position=Vector3([tx, ty, tz]),      # Translation correction
    rotation=Quaternion.from_eulers([rx, ry, rz])  # Rotation correction
)
```

### Frame-Based Corrections

Corrections can vary across a flight. The system supports:

1. **Default correction**: Applied to all frames
2. **Interval corrections**: Different corrections for frame ranges

```json
{
  "corrections": [
    {
      "start frame": 0,
      "end frame": 500,
      "translation": {"x": 0.5, "y": -0.3, "z": 1.2},
      "rotation": {"x": 0, "y": 0, "z": 0.02}
    }
  ],
  "default": {
    "translation": {"x": 0, "y": 0, "z": 0},
    "rotation": {"x": 0, "y": 0, "z": 0}
  }
}
```

---

## Usage

### Docker Deployment

Build the Docker image:
```bash
docker build --tag orthorender -f Dockerfile .
```

Run with CPU:
```bash
docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  --name orthorenderer \
  -e INPUT_DIR="/input" \
  -e OUTPUT_DIR="/output" \
  orthorender
```

Run with GPU acceleration:
```bash
docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  --ipc=host --gpus '"device=0"' \
  --name orthorenderer \
  -e INPUT_DIR="/input" \
  -e OUTPUT_DIR="/output" \
  -e ALFS_DEVICE="cuda" \
  orthorender
```

Note the image ships the **CPU** torch build by default, because the CUDA wheel adds roughly
2 GB. For GPU rendering, drop the `--index-url .../whl/cpu` line from the `Dockerfile` and
rebuild, or base the image on an `nvidia/cuda` runtime image.

The image no longer installs `libx11`, `libxrandr`, `libxinerama`, `libxi`, `mesa` or
`xvfb`, and the entry point no longer starts a virtual display before sleeping two seconds
and hoping. The renderer never opens a GL context, so the ModernGL/Xvfb artifacts described
in the previous version's *Known Limitations* no longer apply.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | - | Path to input dataset folder |
| `OUTPUT_DIR` | - | Path to output folder |
| `SPLITS` | `train,val,test` | Comma-separated list of splits to process |
| `CAMERA_DISTANCE` | `10.0` | Camera height above terrain (meters) |
| `ORTHO_WIDTH` | `70` | Orthographic frustum width (meters) |
| `ORTHO_HEIGHT` | `70` | Orthographic frustum height (meters) |
| `INPUT_WIDTH` | `1024` | Input image width (pixels) |
| `INPUT_HEIGHT` | `1024` | Input image height (pixels) |
| `RENDER_WIDTH` | `2048` | Output image width (pixels) |
| `RENDER_HEIGHT` | `2048` | Output image height (pixels) |
| `INITIAL_SKIP` | `0` | Frames to skip at start |
| `ADD_BACKGROUND` | `1` | Overlay result on DEM render |
| `FOVY` | `50.0` | Field of view for perspective camera |
| `ASPECT_RATIO` | `1.0` | Camera aspect ratio |
| `SAVE_LABELED_IMAGES` | `0` | Save images with drawn labels |
| `PROJECT_ORTHOGONAL` | `1` | Use orthographic (1) or ALFS (0) mode |
| `ADDITIONAL_ROTATIONS` | `0` | Extra rotated views per frame |
| `ROTATION_LIMIT` | `2π` | Max rotation angle for augmentation |
| `ROTATION_SEED` | `-1` | Random seed (-1 for random) |
| `MERGE_LABELS_IN_ALFS` | `1` | Merge labels from all ALFS frames |
| `APPLY_NMS` | `0` | Apply non-maximum suppression |
| `NMS_IOU` | `0.9` | NMS IoU threshold |
| `IS_THERMAL` | `1` | Process thermal (1) or RGB (0) data |

### Input Data Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── {flight_id}_{frame_id}.png
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── {flight_id}_{frame_id}.txt  (YOLO format)
│   │   └── ...
│   ├── val/
│   └── test/
├── correction_data/
│   ├── {flight_id}_dem.glb            # Digital Elevation Model
│   ├── {flight_id}_matched_poses.json # Camera poses
│   ├── {flight_id}_correction.json    # GPS/IMU corrections
│   ├── {flight_id}_mask_t.png         # Thermal mask
│   └── {flight_id}_mask_r.png         # RGB mask
├── export_train.json
├── export_val.json
└── export_test.json
```

---

## Known Limitations

1. **~~Virtual Frame Buffer Artifacts~~ — resolved.** The ModernGL/Xvfb transparency
   artifacts are gone: there is no GL context, no driver-owned surface and no shared
   framebuffer binding. Accumulators are tensors allocated per render call.

   The port also found a **second, host-side cause** of the same symptom that was present on
   every platform: `render_integral` called `np.divide(..., where=mask)` without `out=`, so
   every pixel below the alpha threshold held uninitialised heap memory rather than zeros.
   That is fixed here, and the one-line fix is worth backporting to `alfs_py`. See
   [`docs/MIGRATION_REVIEW.md` §3.1](docs/MIGRATION_REVIEW.md).

2. **CPU throughput.** A pure-torch rasteriser is slower than a GPU driver. A 30-shot ALFS
   integral at 2048×2048 over a 131 k-triangle DEM takes about 7 s per output frame on CPU.
   Use `device="cuda"` for production volumes; benchmark with
   `python -m test.bench.bench_raster`.

3. **DEM Resolution**: Accuracy of geo-referencing depends on DEM resolution. High-resolution DEMs are recommended for best results.

4. **Correction Determination**: Manual correction factors must be determined per-flight through visual inspection of static object alignment.

5. **~~Latent defects in the coordinate-conversion helpers~~ — resolved.** A batch of
   long-standing bugs in `convert.py` and the label camera has been fixed in **both** this
   project and `alfs_py`: the transposed ray rotation and its Euler-negating compensator, the
   `world_to_pixel_coord` broadcast that only accepted 1 or 4 points, the never-working
   orthographic ray branch, and the non-invertible `change_pixel_origin`. See
   [`docs/MIGRATION_REVIEW.md` §9](docs/MIGRATION_REVIEW.md).

---

## Acknowledgments

This project is funded by the Austrian Research Promotion Agency FFG (project THUMPER; program number: 917796) and was developed as part of the BAMBI research project (program number: 892231).
