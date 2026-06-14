"""Legacy HRNet non-elephant pseudo-label script.

This file used to execute pseudo-label generation at import time with local
paths and empty checkpoint placeholders. It is intentionally kept as a guarded
compatibility entry point so users do not accidentally run a broken workflow.

Use the supported ViTPose demo and pseudo-label scripts under `models/ViTPose`
or `scripts/quadpose_vitpose_demo.py` for new work.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    raise SystemExit(
        "models/HRNet/pseudo_others_code.py is a legacy script and is not "
        "production-ready. Use scripts/quadpose_vitpose_demo.py for inference "
        "or port this workflow to an argument-driven pseudo-label CLI."
    )


if __name__ == "__main__":
    main()
