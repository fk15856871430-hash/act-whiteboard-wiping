#!/usr/bin/env python3
"""Create a standalone copy of the implemented whiteboard workflow (no external writes)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
LEROBOT_SOURCE = Path("/home/user/catkin_ws_ENCOS擦白板/catkin_ws_ENCOS/lerobot")
CONDA = Path("/home/user/miniconda3")
DATASET = Path("/mnt/bigdrive/datasets/wipe_board_act_lerobot_v3")
MODELS = Path("/mnt/bigdrive/output_models/wipe_board_act")
SIM = Path("/home/user/isaacsim")


def command(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def output(args, **kwargs):
    return command(args, stdout=subprocess.PIPE, **kwargs).stdout.decode()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def copy_tree(source, target, excludes=()):
    target.mkdir(parents=True, exist_ok=True)
    command(["rsync", "-a", "--info=stats1", *[f"--exclude={x}" for x in excludes],
             str(source) + "/", str(target) + "/"])


def copy_source(source, target, extras=()):
    paths = set(output(["git", "-C", source, "ls-files", "-z"]).split("\0")) - {""}
    for extra in extras:
        item = source / extra
        if not item.exists():
            raise FileNotFoundError(item)
        if item.is_dir():
            paths.update(str(p.relative_to(source)) for p in item.rglob("*") if p.is_file())
        else:
            paths.add(extra)
    paths = sorted(p for p in paths if "__pycache__" not in Path(p).parts
                   and not p.endswith((".pyc", ".zip")) and "local.env" != Path(p).name)
    target.mkdir(parents=True, exist_ok=True)
    command(["rsync", "-a", "--from0", "--files-from=-", str(source) + "/", str(target) + "/"],
            input=("\0".join(paths) + "\0").encode())
    return len(paths)


ENV_QUERY = r'''
import importlib.metadata as m, json, pathlib, sys
root = pathlib.Path(sys.prefix).resolve()
result = []
for d in m.distributions():
    p = pathlib.Path(d.locate_file('')).resolve()
    if not p.is_relative_to(root):
        continue
    direct = d.read_text('direct_url.json')
    result.append({'name': d.metadata['Name'], 'version': d.version,
                   'direct_url': json.loads(direct) if direct else None})
print(json.dumps({'python': sys.version, 'prefix': sys.prefix, 'packages': result}))
'''


def snapshot(kit):
    folder = kit / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "repositories": {}}
    for label, repo in [("IsaacLab", ROOT), ("lerobot", LEROBOT_SOURCE)]:
        provenance["repositories"][label] = {
            "original_path": str(repo), "commit": output(["git", "-C", repo, "rev-parse", "HEAD"]).strip(),
            "status": output(["git", "-C", repo, "status", "--short"])}
        (folder / f"{label}.patch").write_text(output(["git", "-C", repo, "diff", "--binary", "HEAD"]))
    provenance["os"] = Path("/etc/os-release").read_text()
    provenance["gpu"] = output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"])
    provenance["isaacsim_version"] = (SIM / "VERSION").read_text().strip()
    provenance["last_checkpoint"] = str((MODELS / "checkpoints/last").resolve())
    dump(folder / "provenance.json", provenance)
    clean_env = dict(os.environ)
    clean_env.pop("PYTHONPATH", None)  # Never export injected ROS Python 3.10 packages.
    for name in ["env_isaaclab", "lerobot"]:
        prefix = CONDA / "envs" / name
        (folder / f"{name}.conda-explicit.txt").write_text(output([CONDA / "bin/conda", "list", "-p", prefix, "--explicit"]))
        info = json.loads(output([prefix / "bin/python", "-c", ENV_QUERY], env=clean_env))
        dump(folder / f"{name}.packages.json", info)
        pins = [f"{p['name']}=={p['version']}" for p in info["packages"]
                if not p["direct_url"] or not p["direct_url"].get("dir_info", {}).get("editable")]
        (folder / f"{name}.constraints.txt").write_text("\n".join(sorted(set(pins), key=str.lower)) + "\n")
    files = sorted((ROOT / "vla_dataset").glob("*.hdf5"))
    dump(folder / "raw_episodes.json", [{"name": p.name, "size_bytes": p.stat().st_size,
          "in_frozen_777": i < 777} for i, p in enumerate(files)])
    if len(files) < 777:
        raise ValueError("Original raw dataset has fewer than 777 files")
    for name, source in [("dataset_info.json", DATASET / "meta/info.json"),
                         ("model_config.json", MODELS / "checkpoints/last/pretrained_model/config.json"),
                         ("train_config.original.json", MODELS / "checkpoints/last/pretrained_model/train_config.json")]:
        shutil.copy2(source, folder / name)


def checksum_tree(bundle):
    records = []
    last_report = time.monotonic()
    total = 0
    with (bundle / "SHA256SUMS").open("w") as sums:
        for path in sorted(bundle.rglob("*")):
            rel = path.relative_to(bundle).as_posix()
            if rel in {"SHA256SUMS", "bundle_manifest.json"}:
                continue
            if path.is_symlink():
                records.append({"path": rel, "symlink": os.readlink(path)})
                continue
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while block := handle.read(8 * 1024 * 1024):
                    digest.update(block)
            size = path.stat().st_size
            total += size
            value = digest.hexdigest()
            sums.write(f"{value}  {rel}\n")
            records.append({"path": rel, "size_bytes": size, "sha256": value})
            if time.monotonic() - last_report > 20:
                print(f"Hashed {total / 1024**3:.1f} GiB ({len(records)} files)", flush=True)
                last_report = time.monotonic()
    dump(bundle / "bundle_manifest.json", {"files": records, "total_bytes": total,
         "created_utc": datetime.now(timezone.utc).isoformat()})
    print(f"READY: {bundle} ({total / 1024**3:.2f} GiB, {len(records)} entries)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["demo", "full"], default="full")
    parser.add_argument("--include-sim", action="store_true")
    parser.add_argument("--rehash", action="store_true", help="Only regenerate checksums of an existing bundle")
    args = parser.parse_args()
    bundle = args.output.resolve()
    if args.rehash:
        if not (bundle / "IsaacLab/whiteboard_repro/README.md").is_file():
            raise ValueError("Not a whiteboard bundle")
        checksum_tree(bundle)
        return
    if bundle.exists():
        raise FileExistsError(f"Use a new empty output path: {bundle}")
    if bundle.is_relative_to(ROOT) or bundle.is_relative_to(LEROBOT_SOURCE):
        raise ValueError("Output must be outside either source repository")
    minimum_gb = (85 if args.include_sim else 65) if args.profile == "full" else 2
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(bundle.parent).free < minimum_gb * 1024**3:
        raise RuntimeError(f"At least {minimum_gb} GiB free space required")
    bundle.mkdir()
    print(f"Creating {args.profile} bundle: {bundle}", flush=True)
    copy_source(ROOT, bundle / "IsaacLab", ["deploy_act.sh", "train_act.sh", "ACT_WIPE_BOARD.md",
                "official_act", "policy_servers/__init__.py", "policy_servers/lerobot_policy_server.py",
                "policy_servers/websocket_server.py", "policy_servers/requirements-lerobot.txt",
                "scripts/dual_arm", "scripts/my_projects/dagger728.py", "scripts/my_projects/wipe728.py",
                "scripts/my_projects/planner728.py", "scripts/export_act_wipe_csv.py",
                "scripts/plot_act_wipe_trajectory.py", "whiteboard_repro"])
    # Include the actual editable source plus all its local custom robot modules; not output archives.
    extras = output(["git", "-C", LEROBOT_SOURCE, "ls-files", "--others", "--exclude-standard", "-z"]).split("\0")
    extras = [p for p in extras if p and not p.endswith(".zip") and not p.startswith(("outputs/", "data/"))]
    copy_source(LEROBOT_SOURCE, bundle / "lerobot", extras)
    snapshot(bundle / "IsaacLab/whiteboard_repro")
    # Also make the frozen list usable from the current repository.
    copy_tree(bundle / "IsaacLab/whiteboard_repro/snapshots", ROOT / "whiteboard_repro/snapshots")
    copy_tree(DATASET / "meta", bundle / "data/wipe_board_act_lerobot_v3/meta")
    if args.profile == "full":
        copy_tree(DATASET / "data", bundle / "data/wipe_board_act_lerobot_v3/data")
        copy_tree(ROOT / "vla_dataset", bundle / "data/raw_hdf5", ["__pycache__"])
        copy_tree(MODELS, bundle / "models/wipe_board_act")
        copy_tree(ROOT / "scripts/my_projects", bundle / "archive/my_projects", ["__pycache__"])
        copy_tree(ROOT / "dual_arm", bundle / "archive/robot_source")
        copy_tree(ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/whiteboard_erasing",
                  bundle / "archive/whiteboard_erasing_rl", ["__pycache__"])
    else:
        # Dereference 'last' by copying its resolved directory, preserving training_state as well.
        latest = (MODELS / "checkpoints/last").resolve(strict=True)
        dest = bundle / "models/wipe_board_act/checkpoints" / latest.name
        copy_tree(latest, dest)
        (dest.parent / "last").symlink_to(latest.name, target_is_directory=True)
    weight = Path("/mnt/bigdrive/torch_cache/hub/checkpoints/resnet18-f37072fd.pth")
    target = bundle / "cache/torch/hub/checkpoints"
    target.mkdir(parents=True)
    shutil.copy2(weight, target / weight.name)
    if args.include_sim:
        # These are session output dirs, not runtime assets. Avoid inaccessible screenshots.
        copy_tree(SIM, bundle / "isaacsim", ["kit/data", "kit/logs", "kit/cache", "*.log", "__pycache__"])
        self_link = bundle / "isaacsim/isaacsim"
        if self_link.is_symlink() and self_link.resolve() == SIM.resolve():
            self_link.unlink()
            self_link.symlink_to(".", target_is_directory=True)
        (bundle / "IsaacLab/_isaac_sim").symlink_to("../isaacsim", target_is_directory=True)
    shutil.copy2(ROOT / "whiteboard_repro/README.md", bundle / "README.md")
    examples = bundle / "examples"
    examples.mkdir()
    for name in ["act_wipe_episode.json", "act_wipe_episode_curves"]:
        source = ROOT / "trajectory_results" / name
        if source.is_dir():
            copy_tree(source, examples / name)
        elif source.is_file():
            shutil.copy2(source, examples / name)
    sample = examples / "act_wipe_episode.json"
    if sample.is_file() and json.loads(sample.read_text()).get("schema_version") == 5:
        command([sys.executable, ROOT / "scripts/export_act_wipe_csv.py",
                 sample, examples / "act_wipe_episode.csv"])
    dump(bundle / "build_options.json", {"profile": args.profile, "include_sim": args.include_sim,
         "python_envs": "packed separately; see README", "room_assets": "online unless WHITEBOARD_ROOM_USD supplied"})
    checksum_tree(bundle)


if __name__ == "__main__":
    main()
