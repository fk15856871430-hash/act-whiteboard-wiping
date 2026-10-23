#!/usr/bin/env python3
"""Download public release pieces anonymously, resume transfers and verify before extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'act-whiteboard-reproduction', **(headers or {})}), timeout=90)


def download(url, target, size=None):
    partial = target.with_name(target.name + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    if size is not None and offset == size:
        partial.rename(target)
        return
    with request(url, {'Range': f'bytes={offset}-'} if offset else {}) as response:
        append = offset > 0 and response.status == 206
        if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise ValueError('Unexpected resume range')
        with partial.open('ab' if append else 'wb') as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
    if size is not None and partial.stat().st_size != size:
        raise ValueError(f'Incomplete download: {partial}; rerun to resume')
    partial.rename(target)


def verify(path, item):
    if path.stat().st_size != item['size']:
        raise ValueError(f'Wrong file size: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != item['sha256']:
        raise ValueError(f'SHA256 mismatch: {path}; move that file aside and download it again')


def extract(paths, destination):
    # Validate archive member paths/links before allowing tar to write. Two streaming passes
    # avoid allocating an extra full-size tar file on disk.
    import tarfile
    import posixpath

    def stream_to(proc):
        try:
            for path in paths:
                with path.open('rb') as stream:
                    while block := stream.read(8 * 1024 * 1024):
                        proc.stdin.write(block)
        except BaseException:
            proc.stdin.close()
            proc.wait()
            raise
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('tar extraction failed')

    class Joined:
        def __init__(self):
            self.files = iter(paths)
            self.current = None
        def read(self, size):
            out = bytearray()
            while len(out) < size:
                if self.current is None:
                    try:
                        self.current = next(self.files).open('rb')
                    except StopIteration:
                        break
                block = self.current.read(size - len(out))
                if block:
                    out.extend(block)
                else:
                    self.current.close()
                    self.current = None
            return bytes(out)
        def close(self):
            if self.current:
                self.current.close()

    joined = Joined()
    try:
        with tarfile.open(fileobj=joined, mode='r|') as archive:
            for member in archive:
                name = posixpath.normpath(member.name)
                if name.startswith('/') or name == '..' or name.startswith('../'):
                    raise ValueError(f'Unsafe archive path: {member.name}')
                resolved = (destination / name).resolve()
                if not resolved.is_relative_to(destination.resolve()):
                    raise ValueError(f'Archive path escapes destination: {name}')
                if member.isdev() or member.isfifo():
                    raise ValueError(f'Unsupported archive member: {name}')
                if member.issym() or member.islnk():
                    parent = posixpath.dirname(name) if member.issym() else ''
                    link = posixpath.normpath(posixpath.join(parent, member.linkname))
                    if link.startswith('/') or link == '..' or link.startswith('../'):
                        raise ValueError(f'Unsafe archive link: {name}')
    finally:
        joined.close()
    proc = subprocess.Popen(['tar', '-xf', '-', '--no-same-owner', '-C', str(destination)], stdin=subprocess.PIPE)
    stream_to(proc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='fk15856871430-hash/act-whiteboard-wiping')
    p.add_argument('--tag', default='v1.0.0')
    p.add_argument('--groups', nargs='+', default=['robot-assets', 'runtime', 'environments'])
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--download-only', action='store_true')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo) or not re.fullmatch(r'[A-Za-z0-9_.-]+', args.tag):
        p.error('Invalid repository or release tag')
    root = args.root.resolve()
    cache = root / 'downloads' / args.tag
    cache.mkdir(parents=True, exist_ok=True)
    with request(f'https://api.github.com/repos/{args.repo}/releases/tags/{args.tag}') as response:
        release = json.load(response)
    urls = {a['name']: a['browser_download_url'] for a in release['assets']}
    with request(urls['assets.json']) as response:
        manifest = json.load(response)
    for name in args.groups:
        group = manifest['groups'][name]
        paths = []
        for item in group['parts']:
            if Path(item['name']).name != item['name']:
                raise ValueError('Invalid asset name')
            target = cache / item['name']
            if not target.exists():
                print(f'Downloading {target.name}', flush=True)
                download(urls[item['name']], target, item['size'])
            verify(target, item)
            paths.append(target)
        if not args.download_only:
            stamp = cache / f'{name}.extracted.json'
            expected = json.dumps(group, sort_keys=True)
            if stamp.exists() and stamp.read_text() == expected and all((root / n).exists() for n in group['paths']):
                print(f'Already extracted: {name}', flush=True)
                continue
            for n in group['paths']:
                if (root / n).exists():
                    raise FileExistsError(f'Refusing to overwrite existing data: {root / n}')
            print(f'Extracting {name}', flush=True)
            extract(paths, root)
            stamp.write_text(expected)
    print('Done. Download parts are kept in downloads/ for reuse; they are not Git files.')


if __name__ == '__main__':
    main()
