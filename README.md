# HapticLens

*Interactive Vibrotactile Haptic Generation from Spatially Localized Video Motion*

![HapticLens overview](hapticlens-overview.png)

## Requirements

- Python 3.11
- CUDA-capable GPU for the CV pipelines (VRAM requirements depend on the input video resolution and number of frames)
- `ffmpeg` and `ffprobe`
- Quest controller playback support is available via the companion [questxr-happlay](https://github.com/kevin-cgc/questxr-happlay) project

## Setup

```bash
uv sync
```

If `ffmpeg` or `ffprobe` are not on `PATH`, set:

```bash
FFMPEG_PATH=/path/to/ffmpeg
FFPROBE_PATH=/path/to/ffprobe
```

## Quick Start

Interactive GUI:

```bash
uv run src/gui.py /path/to/video.mp4 1
```

There is also a primitive batch processing script, that uses fixed extraction regions:

```bash
uv run src/create-dataset-batch.py \
  --input-dir /path/to/input_videos \
  --output-dir /path/to/output_wavs \
  --algorithm phase \
  --extraction-percentages 1.0 0.5 0.25 0.1
```

## ACM Reference
```
Kevin John and Hasti Seifi. 2026. HapticLens: Interactive Vibrotactile Haptic Generation from Spatially Localized Video Motion. In Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI ’26), April 13–17, 2026, Barcelona, Spain. ACM, New York, NY, USA, 21 pages. https://doi.org/10.1145/3772318.3790269
```