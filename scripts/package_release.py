#!/usr/bin/env python3
"""Create a source-only archive from an explicit allowlist; never replace an archive."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from verify_release import ROOT, public_files, verify


def package(output):
    verify()
    paths = public_files()
    payload = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, 'imagenet-fid/' + path.relative_to(ROOT).as_posix())
        archive.writestr('imagenet-fid/RELEASE_SHA256SUMS.json', json.dumps(payload, indent=2) + '\n')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive integrity check failed')
    print(f'Packaged {len(paths)} files: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    package(parser.parse_args().output)
