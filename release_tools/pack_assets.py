#!/usr/bin/env python3
"""Stream release data into <2 GiB tar pieces, with SHA256 and a download manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

GROUPS = {
    'robot-assets': ['IsaacLab/scripts/dual_arm'],
    'runtime': ['models', 'cache/torch', 'data/wipe_board_act_lerobot_v3/meta'],
    'environments': ['environments'],
    'training-data': ['data/wipe_board_act_lerobot_v3/data'],
    'raw-demonstrations': ['data/raw_hdf5'],
}
PIECE_SIZE = 1800 * 1024 * 1024


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def pack(source, dest, group):
    summary = dest / f'{group}.json'
    if summary.exists():
        record = json.loads(summary.read_text())
        for item in record['parts']:
            path = dest / item['name']
            if path.stat().st_size != item['size'] or digest(path) != item['sha256']:
                raise ValueError(f'Existing part failed verification: {path}')
        print(f'Verified existing group: {group}', flush=True)
        return record
    if list(dest.glob(f'{group}.tar.part*')):
        raise FileExistsError(f'Incomplete group {group}; use a new output directory or move its old parts aside')
    for name in GROUPS[group]:
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
    proc = subprocess.Popen(['tar', '-cf', '-', '-C', str(source), '--', *GROUPS[group]], stdout=subprocess.PIPE)
    parts = []
    last_report = time.monotonic()
    try:
        while block := proc.stdout.read(min(8 * 1024 * 1024, PIECE_SIZE)):
            name = f'{group}.tar.part{len(parts) + 1:04d}'
            hashed, count = hashlib.sha256(), 0
            with (dest / name).open('xb') as output:
                while block:
                    output.write(block)
                    hashed.update(block)
                    count += len(block)
                    if count == PIECE_SIZE:
                        break
                    block = proc.stdout.read(min(8 * 1024 * 1024, PIECE_SIZE - count))
            parts.append({'name': name, 'size': count, 'sha256': hashed.hexdigest()})
            if time.monotonic() - last_report > 20:
                print(f'{group}: {sum(x["size"] for x in parts) / 1024**3:.1f} GiB packaged', flush=True)
                last_report = time.monotonic()
        if proc.wait() != 0:
            raise RuntimeError(f'tar failed: {group}')
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        proc.stdout.close()
    record = {'paths': GROUPS[group], 'parts': parts}
    summary.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Completed {group}: {len(parts)} pieces', flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    args = p.parse_args()
    source, dest = args.bundle.resolve(), args.output.resolve()
    if not (source / 'bundle_manifest.json').is_file():
        raise ValueError('Expected a completed migration bundle')
    if dest.is_relative_to(source):
        raise ValueError('Release output must be outside the migration bundle')
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / 'assets.json'
    records = json.loads(manifest_path.read_text())['groups'] if manifest_path.exists() else {}
    records.update({g: pack(source, dest, g) for g in args.groups})
    (dest / 'assets.json').write_text(json.dumps({'schema': 1, 'groups': records}, indent=2) + '\n')
    (dest / 'SHA256SUMS').write_text(''.join(f'{x["sha256"]}  {x["name"]}\n'
        for group in records.values() for x in group['parts']))
    print(f'READY: {dest}', flush=True)


if __name__ == '__main__':
    main()
