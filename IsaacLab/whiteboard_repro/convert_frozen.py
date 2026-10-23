"""Convert the frozen original 777 filenames without mixing newly collected data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args, extra = parser.parse_known_args()
    kit = Path(__file__).resolve().parent
    entries = json.loads((kit / "snapshots/raw_episodes.json").read_text())
    names = [row["name"] for row in entries if row["in_frozen_777"]]
    if len(names) != 777:
        raise ValueError(f"Expected frozen 777 files, got {len(names)}")
    with tempfile.TemporaryDirectory(prefix="whiteboard_frozen_") as folder:
        for name in names:
            source = (args.input_dir / name).resolve(strict=True)
            (Path(folder) / name).symlink_to(source)
        subprocess.run([sys.executable, str(kit.parent / "official_act/convert_dataset.py"),
                        "--input_dir", folder, "--output_path", str(args.output_path),
                        "--repo_id", "wipe_board_act_lerobot_v3", "--episode_count", "777",
                        "--image_size", "256", "--fps", "50", *extra], check=True)


if __name__ == "__main__":
    main()
