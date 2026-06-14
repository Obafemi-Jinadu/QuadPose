"""Regression checks for released QuadPose checkpoint compatibility."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAD_FILE = (
    REPO_ROOT
    / "models"
    / "ViTPose"
    / "mmpose"
    / "models"
    / "heads"
    / "topdown_heatmap_simple_head.py"
)


def test_other_animals_legacy_name_is_remapped():
    source = HEAD_FILE.read_text(encoding="utf-8")

    assert "final_layer_otherAnimals_femi_edited" in source
    assert "final_layer_otherAnimals" in source
    assert "def _load_from_state_dict" in source


def test_legacy_key_rewrite_behavior():
    legacy = "keypoint_head.final_layer_otherAnimals_femi_edited.weight"
    current = "keypoint_head.final_layer_otherAnimals.weight"
    state_dict = {legacy: object()}

    for key in list(state_dict.keys()):
        if key.startswith("keypoint_head.final_layer_otherAnimals_femi_edited."):
            new_key = "keypoint_head.final_layer_otherAnimals." + key[
                len("keypoint_head.final_layer_otherAnimals_femi_edited.") :
            ]
            if new_key not in state_dict:
                state_dict[new_key] = state_dict[key]
            del state_dict[key]

    assert current in state_dict
    assert legacy not in state_dict
