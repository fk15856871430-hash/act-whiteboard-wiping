"""Read-only checks; a successful check does not replace a rendered simulation run."""
import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["policy", "sim"], required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    errors = []
    def require(condition, message):
        print(("OK   " if condition else "FAIL ") + message)
        if not condition:
            errors.append(message)
    print(f"[{args.role}] Python {sys.version.split()[0]}: {sys.executable}")
    modules = (["torch", "torchvision", "lerobot", "websockets", "msgpack_numpy", "h5py", "cv2"]
               if args.role == "policy" else
               ["torch", "scipy", "sklearn", "cv2", "h5py", "zmq", "msgpack_numpy"])
    for module in modules:
        try:
            loaded = importlib.import_module(module)
            require(True, f"{module}: {getattr(loaded, '__version__', 'imported')}")
        except Exception as exc:
            require(False, f"{module}: {exc}")
    import torch
    require(torch.cuda.is_available(), "CUDA available")
    if args.role == "policy":
        require(sys.version_info[:2] == (3, 12), "policy Python 3.12")
        require(metadata.version("lerobot") == "0.6.2", "bundled LeRobot 0.6.2")
        for name in ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
                     "policy_preprocessor_step_3_normalizer_processor.safetensors",
                     "policy_postprocessor_step_0_unnormalizer_processor.safetensors"]:
            require((args.model / name).is_file(), f"checkpoint {name}")
        config = json.loads((args.model / "config.json").read_text())
        require(config["chunk_size"] == 50 and config["n_action_steps"] == 50, "chunk/executed actions 50/50")
        require(config["input_features"]["observation.state"]["shape"] == [20], "20D policy state")
        for name in ["info.json", "stats.json", "tasks.parquet"]:
            require((args.dataset / "meta" / name).is_file(), f"dataset meta/{name}")
        info = json.loads((args.dataset / "meta/info.json").read_text())
        require(info["fps"] == 50, "dataset 50 Hz")
        print("INFO full training dataset present" if (args.dataset / "data").is_dir()
              else "INFO metadata-only dataset: inference available; training requires data/")
    else:
        root = Path(__file__).resolve().parent.parent
        require(sys.version_info[:2] == (3, 11), "simulation Python 3.11")
        try:
            require(metadata.version("isaaclab") == "0.54.3", "IsaacLab installed editable")
        except metadata.PackageNotFoundError:
            require(False, "IsaacLab missing; install local source/isaaclab")
        require((root / "_isaac_sim/VERSION").is_file(), "_isaac_sim symlink and VERSION")
        require((root / "scripts/dual_arm/urdf/dual_arm.usd").is_file(), "local robot USD")
        print("INFO websockets / pxr may be supplied by Isaac Sim after AppLauncher; checked in GUI run.")
        print("INFO Simple_Room still needs online Isaac assets or WHITEBOARD_ROOM_USD with dependencies.")
    if errors:
        raise SystemExit(f"{len(errors)} checks failed")
    print("Checks passed. Next: start server, then rendered scene.")


if __name__ == "__main__":
    main()
