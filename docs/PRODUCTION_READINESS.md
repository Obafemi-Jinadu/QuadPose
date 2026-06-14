# Production Readiness

QuadPose is currently a research release. The repository now has a supported
demo entry point, but it should not be treated as a packaged production service
until the remaining checklist is complete.

## Supported Demo

Run ViTPose inference on a single image:

```bash
python scripts/quadpose_vitpose_demo.py \
  --image path/to/image.jpg \
  --pose-config models/ViTPose/vitpose_base_teacher.py \
  --pose-checkpoint path/to/quadpose_weights.pth \
  --animal-type elephant \
  --device cuda:0 \
  --out-image outputs/demo.jpg \
  --out-json outputs/demo.json
```

Use `--animal-type quadruped` for the non-elephant QuadPose head.

The demo expects the ViTPose/MMDetection runtime to be installed. Model weights
are intentionally not committed to the repository; download them from the link
in the root README and pass the local checkpoint path with `--pose-checkpoint`.

## Remaining Work

- Replace full environment exports with a minimal, tested environment file.
- Add a small sample image and a lightweight smoke test for config loading.
- Convert pseudo-label generation into one supported CLI with explicit input,
  output, config, checkpoint, threshold, and device arguments.
- Remove or archive copied scratch files after confirming they are not needed.
- Add CI that runs formatting, import checks, and the smoke test.
- Publish a versioned release once the install and demo commands are verified
  on a fresh machine.
