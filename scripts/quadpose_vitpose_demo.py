#!/usr/bin/env python3
"""Run QuadPose/ViTPose inference on one image.

This is the supported, top-level demo entry point for the repository. It wraps
the ViTPose + MMDetection inference path with explicit arguments and validation
so users do not need to edit Python files before running a demo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VITPOSE_ROOT = REPO_ROOT / "models" / "ViTPose"


def _require_file(path: str, label: str) -> str:
    if path.startswith(("http://", "https://")):
        return path
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return str(resolved)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run QuadPose/ViTPose top-down pose estimation on one image."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--pose-config", required=True, help="ViTPose config path.")
    parser.add_argument("--pose-checkpoint", required=True, help="ViTPose weights path.")
    parser.add_argument(
        "--det-config",
        default="models/ViTPose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
        help="MMDetection config path.",
    )
    parser.add_argument(
        "--det-checkpoint",
        default=(
            "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/"
            "faster_rcnn_r50_fpn_1x_coco/"
            "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
        ),
        help="MMDetection checkpoint path or URL.",
    )
    parser.add_argument(
        "--animal-type",
        choices=("elephant", "quadruped"),
        default="elephant",
        help="Controls the detector category and QuadPose prediction head.",
    )
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--bbox-thr", type=float, default=0.3, help="Detection score threshold.")
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Visualization keypoint threshold.")
    parser.add_argument("--out-image", required=True, help="Output visualization image path.")
    parser.add_argument("--out-json", help="Optional output JSON path for raw pose results.")
    parser.add_argument("--radius", type=int, default=4, help="Keypoint radius in the output image.")
    parser.add_argument("--thickness", type=int, default=1, help="Skeleton line thickness.")
    return parser


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is required by mmpose at runtime.
        np = None

    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    args = _build_parser().parse_args()

    if not VITPOSE_ROOT.is_dir():
        raise RuntimeError(f"ViTPose source tree not found: {VITPOSE_ROOT}")
    sys.path.insert(0, str(VITPOSE_ROOT))

    image = _require_file(args.image, "image")
    pose_config = _require_file(args.pose_config, "pose config")
    pose_checkpoint = _require_file(args.pose_checkpoint, "pose checkpoint")
    det_config = _require_file(args.det_config, "detector config")
    det_checkpoint = _require_file(args.det_checkpoint, "detector checkpoint")

    try:
        from mmdet.apis import inference_detector, init_detector
        from mmpose.apis import (
            inference_top_down_pose_model_modified,
            init_pose_model,
            process_mmdet_results,
            vis_pose_result_modified,
        )
        from mmpose.datasets import DatasetInfo
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Missing runtime dependencies. Install the ViTPose/MMDetection "
            "environment before running this demo."
        ) from exc

    det_cat_id = 21 if args.animal_type == "elephant" else None
    pose_cat_id = 0 if args.animal_type == "elephant" else 1

    det_model = init_detector(det_config, det_checkpoint, device=args.device.lower())
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn("dataset_info is missing from the pose config.", RuntimeWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    mmdet_results = inference_detector(det_model, image)
    if det_cat_id is None:
        # Use every non-person COCO class for the broad quadruped head. COCO index
        # 0 is person, and index 20 is elephant in the detector output.
        det_results = mmdet_results[0] if isinstance(mmdet_results, tuple) else mmdet_results
        person_results = []
        for class_index, class_boxes in enumerate(det_results):
            if class_index in (0, 20):
                continue
            for bbox in class_boxes:
                person_results.append({"bbox": bbox})
    else:
        person_results = process_mmdet_results(mmdet_results, det_cat_id)

    pose_results, returned_outputs = inference_top_down_pose_model_modified(
        pose_model,
        image,
        person_results,
        bbox_thr=args.bbox_thr,
        format="xyxy",
        dataset=dataset,
        dataset_info=dataset_info,
        cat_id=pose_cat_id,
        return_heatmap=False,
        outputs=None,
    )

    out_image = Path(args.out_image).expanduser()
    if not out_image.is_absolute():
        out_image = REPO_ROOT / out_image
    out_image.parent.mkdir(parents=True, exist_ok=True)

    vis_pose_result_modified(
        pose_model,
        image,
        pose_results,
        dataset=dataset,
        dataset_info=dataset_info,
        kpt_score_thr=args.kpt_thr,
        radius=args.radius,
        thickness=args.thickness,
        show=False,
        out_file=str(out_image),
    )

    if args.out_json:
        out_json = Path(args.out_json).expanduser()
        if not out_json.is_absolute():
            out_json = REPO_ROOT / out_json
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image": image,
            "animal_type": args.animal_type,
            "num_detections": len(person_results),
            "num_poses": len(pose_results),
            "poses": _jsonable(pose_results),
            "outputs": _jsonable(returned_outputs),
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote visualization: {out_image}")
    if args.out_json:
        print(f"Wrote pose JSON: {out_json}")


if __name__ == "__main__":
    main()
