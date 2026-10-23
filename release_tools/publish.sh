#!/usr/bin/env bash
set -euo pipefail
unset GH_DEBUG DEBUG
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo="fk15856871430-hash/act-whiteboard-wiping"
tag="v1.0.0"
assets="${1:?Usage: bash release_tools/publish.sh /absolute/release_assets}"
gh_bin="${GH_BIN:-gh}"
if ! command -v "${gh_bin}" >/dev/null; then gh_bin="${HOME}/.local/bin/gh"; fi
"${gh_bin}" auth status --hostname github.com
login="$("${gh_bin}" api user --jq .login)"
[[ "${login}" == "fk15856871430-hash" ]] || { echo "Wrong GitHub account: ${login}" >&2; exit 1; }
[[ -f "${repo_root}/README.md" && -f "${repo_root}/IsaacLab/whiteboard_repro/run.sh" ]] || exit 1
[[ -f "${assets}/assets.json" ]] || { echo "Run pack_assets.py first" >&2; exit 1; }
# Verify files before any external write. No global git identity or original remote is changed.
(cd "${assets}" && sha256sum -c SHA256SUMS --quiet)
cd "${repo_root}"
if [[ ! -d .git ]]; then git init -b main; fi
if [[ -z "$(git config --local user.name || true)" ]]; then
  git config --local user.name "${login}"
  account_id="$("${gh_bin}" api user --jq .id)"
  git config --local user.email "${account_id}+${login}@users.noreply.github.com"
fi
git add .
if ! git diff --cached --quiet; then git commit -m "Publish ACT whiteboard wiping workflow and reproduction guide"; fi
if ! "${gh_bin}" repo view "${repo}" >/dev/null 2>&1; then
  "${gh_bin}" repo create "${repo}" --public --description "ACT + admittance whiteboard wiping: simulation, collection, training and deployment"
fi
visibility="$("${gh_bin}" api "repos/${repo}" --jq .visibility)"
[[ "${visibility}" == public ]] || { echo "Existing repository is not public; refusing to change visibility" >&2; exit 1; }
if git remote get-url origin >/dev/null 2>&1; then
  [[ "$(git remote get-url origin)" == "https://github.com/${repo}.git" ]] || { echo "Unexpected origin" >&2; exit 1; }
else
  git remote add origin "https://github.com/${repo}.git"
fi
git -c credential.helper= -c "credential.helper=!${gh_bin} auth git-credential" push -u origin main
if ! "${gh_bin}" release view "${tag}" --repo "${repo}" >/dev/null 2>&1; then
  "${gh_bin}" release create "${tag}" --repo "${repo}" --target main --draft \
    --title "ACT whiteboard wiping v1.0.0" \
    --notes "Source, custom robot assets and trajectory examples are in the repository. Download runtime/environments for inference; training-data/raw-demonstrations are optional for retraining. Use release_tools/download_assets.py to verify and extract. Isaac Sim is installed separately from NVIDIA."
fi
# One file at a time. Keep the release draft until every part is uploaded.
# The Python helper verifies existing assets by server SHA256, and never clobbers them.
GH_BIN="${gh_bin}" WB_RELEASE_ASSETS="${assets}" WB_GITHUB_REPO="${repo}" WB_RELEASE_TAG="${tag}" python - <<'PY'
import hashlib,json,os,subprocess,time
from pathlib import Path
gh,repo,tag=os.environ['GH_BIN'],os.environ['WB_GITHUB_REPO'],os.environ['WB_RELEASE_TAG']
root=Path(os.environ['WB_RELEASE_ASSETS']).resolve()
def api():
    return json.loads(subprocess.check_output([gh,'api',f'repos/{repo}/releases/tags/{tag}']))
manifest=json.loads((root/'assets.json').read_text())
items=[x for g in manifest['groups'].values() for x in g['parts']]
for name in ['SHA256SUMS','assets.json']:
    p=root/name
    items.append({'name':name,'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
existing={a['name']:a for a in api()['assets']}
for item in items:
    a=existing.get(item['name'])
    if a:
        if a['size']==item['size'] and a.get('digest')=='sha256:'+item['sha256']:
            print('Already uploaded:',item['name'],flush=True)
            continue
        raise RuntimeError('Existing asset differs or has no verifiable server digest: '+item['name'])
    print('Uploading:',item['name'],flush=True)
    subprocess.run([gh,'release','upload',tag,'--repo',repo,str(root/item['name'])],check=True)
    stored=next(a for a in api()['assets'] if a['name']==item['name'])
    if stored['size']!=item['size'] or stored.get('digest')!='sha256:'+item['sha256']:
        raise RuntimeError('Uploaded asset failed server SHA256 verification: '+item['name'])
PY
release_id="$("${gh_bin}" api "repos/${repo}/releases/tags/${tag}" --jq .id)"
"${gh_bin}" api --method PATCH "repos/${repo}/releases/${release_id}" -F draft=false >/dev/null
echo "Published: https://github.com/${repo}"
echo "Downloads: https://github.com/${repo}/releases/tag/${tag}"
