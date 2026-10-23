"""Replace only the restored environment's machine-specific Isaac activation hook."""
from pathlib import Path
import shlex
import sys

prefix, repo = (Path(p).resolve() for p in sys.argv[1:])
hook = prefix / "etc/conda/activate.d/isaac_sim_env.sh"
if not (prefix / "bin/conda-unpack").is_file():
    raise RuntimeError("Only modify a restored conda-pack environment")
hook.parent.mkdir(parents=True, exist_ok=True)
hook.write_text("#!/usr/bin/env bash\nsource " + shlex.quote(str(repo / "_isaac_sim/setup_conda_env.sh")) + "\n")
print(f"Updated restored Isaac activation hook: {hook}")
