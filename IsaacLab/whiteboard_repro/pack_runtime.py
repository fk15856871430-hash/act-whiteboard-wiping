"""Pack actual runtime files, auditing the two known pip-over-conda replacements first."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from conda_pack import CondaEnv, File


QUERY = r'''
import importlib.metadata as m, json, pathlib
result={}
for name in ['packaging', 'setuptools']:
 d=m.distribution(name)
 missing=[str(f) for f in d.files if not str(f).endswith('.pyc') and not d.locate_file(f).exists()]
 result[name]={'version':d.version,'installer':(d.read_text('INSTALLER') or '').strip(),'missing_record_files':missing}
print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    clean = dict(os.environ)
    clean.pop("PYTHONPATH", None)
    runtime = json.loads(subprocess.check_output([str(prefix / "bin/python"), "-c", QUERY], env=clean))
    expected = {"env_isaaclab": {"packaging": "23.2", "setuptools": "81.0.0"},
                "lerobot": {"packaging": "25.0", "setuptools": "80.10.2"}}
    replaced_targets, replaced_caches, excluded_meta = set(), [], set()
    report = []
    for record in sorted((prefix / "conda-meta").glob("*.json")):
        info = json.loads(record.read_text())
        missing = [f for f in info.get("files", []) if not (prefix / f).exists() and not f.endswith(".pyc")]
        if not missing:
            continue
        name = info["name"]
        if name not in expected.get(prefix.name, {}):
            raise RuntimeError(f"Unaudited missing files in {name}: {missing[:10]}")
        actual = runtime[name]
        if (actual["version"] != expected[prefix.name][name] or actual["installer"] != "pip"
                or actual["missing_record_files"]):
            raise RuntimeError(f"Replacement is not the verified runtime version: {name}: {actual}")
        replaced_targets.update(info["files"])
        if info.get("link", {}).get("source"):
            replaced_caches.append(str(Path(info["link"]["source"]).resolve()) + os.sep)
        excluded_meta.add(f"conda-meta/{record.name}")
        report.append({"package": name, "obsolete_conda_version": info["version"],
                       "runtime_pip_version": actual["version"], "obsolete_missing_files": missing,
                       "action": "pack actual installed pip files; omit obsolete conda record"})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    # Missing files are allowed ONLY after the audit above accepts the known replacements.
    # All replacement files must come from the running env, never the older Conda package cache.
    environment = CondaEnv.from_prefix(str(prefix), ignore_editable_packages=True,
                                       ignore_missing_files=bool(report))
    files = []
    for file in environment.files:
        if file.target in excluded_meta:
            continue
        if file.target in replaced_targets or any(file.source.startswith(p) for p in replaced_caches):
            actual = prefix / file.target
            if not actual.exists():
                raise FileNotFoundError(actual)
            file = File(str(actual), file.target, is_conda=False, file_mode="unknown")
        files.append(file)
    args.output.with_suffix(".audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Packing {prefix} -> {args.output}", flush=True)
    CondaEnv(str(prefix), files).pack(output=str(args.output), format="tar.gz", compress_level=1,
                                     n_threads=4, verbose=False)
    print(f"Packed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
