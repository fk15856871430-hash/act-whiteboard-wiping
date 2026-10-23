#!/usr/bin/env python3
"""Small integration checks for release splitting, hashes and safe extraction."""
import importlib.util
from pathlib import Path
import tempfile


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    pack, download = load('pack_assets'), load('download_assets')
    assert set(pack.GROUPS) == {'robot-assets', 'runtime', 'environments'}
    assert all('/raw_hdf5' not in path and not path.endswith('/data')
               for paths in pack.GROUPS.values() for path in paths)
    with tempfile.TemporaryDirectory(prefix='whiteboard-release-check-') as name:
        root = Path(name)
        source, parts, restored = root/'source', root/'parts', root/'restored'
        for p in [source/'models/checkpoints/100000', source/'cache/torch',
                  source/'data/wipe_board_act_lerobot_v3/meta', parts, restored]:
            p.mkdir(parents=True)
        content = bytes(range(256))*300
        weight = source/'models/checkpoints/100000/weights.bin'
        weight.write_bytes(content)
        (weight.parent.parent/'last').symlink_to('100000', target_is_directory=True)
        (source/'data/wipe_board_act_lerobot_v3/meta/info.json').write_text('{}')
        pack.PIECE_SIZE = 16*1024
        record = pack.pack(source, parts, 'runtime')
        assert len(record['parts']) > 2
        for item in record['parts']:
            download.verify(parts/item['name'], item)
        download.extract([parts/x['name'] for x in record['parts']], restored)
        assert (restored/'models/checkpoints/last/weights.bin').read_bytes() == content
        assert (restored/'models/checkpoints/last').readlink() == Path('100000')
        assert pack.pack(source, parts, 'runtime') == record
        corrupt = parts/record['parts'][0]['name']
        corrupt.write_bytes(b'x'*corrupt.stat().st_size)
        try:
            download.verify(corrupt, record['parts'][0])
        except ValueError:
            pass
        else:
            raise AssertionError('Corruption went undetected')
    print('PASS: multi-part archive, SHA256, relative symlink, repeat verification and corruption detection')


if __name__ == '__main__':
    main()
