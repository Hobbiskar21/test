"""Clear exported test outputs for batsman or bowler runs.

Run from cricket_bowling_analysis:
    python clear.py
"""

import os
import shutil
import stat
from pathlib import Path


OUTPUT_FOLDERS = {
    "batsman": Path("outputs/test_output/batsman"),
    "bowler": Path("outputs/test_output/bowling"),
}

SUBFOLDERS = {
    "batsman": ("video", "keypoints_csv", "metrics_csv", "phases_csv", "analysis_csv"),
    "bowler": ("video", "phase_label_csv", "keypoints_csv", "features_csv", "metrics_csv"),
}


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _on_rm_error(func, path, exc_info):
    exc_type, _, _ = exc_info
    if exc_type is PermissionError:
        os.chmod(path, stat.S_IWRITE)
        func(path)
        return
    raise


def _safe_output_path(target: str) -> Path:
    root = workspace_root().resolve()
    output_path = (root / OUTPUT_FOLDERS[target]).resolve()
    test_output_root = (root / "outputs" / "test_output").resolve()

    if test_output_root not in output_path.parents:
        raise RuntimeError(f"Refusing to clear path outside outputs/test_output: {output_path}")
    return output_path


def clear_output(target: str) -> Path:
    output_path = _safe_output_path(target)
    if output_path.exists():
        shutil.rmtree(output_path, onerror=_on_rm_error)

    for folder in SUBFOLDERS[target]:
        (output_path / folder).mkdir(parents=True, exist_ok=True)

    return output_path


def clear_all_outputs() -> list[Path]:
    cleared_paths = []
    for target in ("bowler", "batsman"):
        cleared_paths.append(clear_output(target))

    root_metrics = (workspace_root() / "outputs" / "test_output" / "metrics_csv").resolve()
    test_output_root = (workspace_root() / "outputs" / "test_output").resolve()
    if test_output_root in root_metrics.parents and root_metrics.exists():
        shutil.rmtree(root_metrics, onerror=_on_rm_error)

    return cleared_paths


def choose_target() -> str:
    print("\nWhat do you want to clear?")
    print("  1. bowler")
    print("  2. batsman")
    print("  3. all")

    choices = {
        "1": "bowler",
        "2": "batsman",
        "3": "all",
        "bowler": "bowler",
        "batsman": "batsman",
        "all": "all",
    }
    while True:
        choice = input("Enter choice: ").strip().lower()
        if choice in choices:
            return choices[choice]
        print("Please enter 1, 2, 3, bowler, batsman, or all.")


def main() -> None:
    target = choose_target()
    if target == "all":
        output_path = (workspace_root() / "outputs" / "test_output").resolve()
    else:
        output_path = _safe_output_path(target)

    confirm = input(f"Clear {target} outputs at {output_path}? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("Clear cancelled.")
        return

    if target == "all":
        cleared_paths = clear_all_outputs()
        for cleared_path in cleared_paths:
            print(f"Cleared output folder: {cleared_path}")
        return

    cleared_path = clear_output(target)
    print(f"Cleared {target} outputs: {cleared_path}")


if __name__ == "__main__":
    main()
