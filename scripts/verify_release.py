#!/usr/bin/env python3
"""Verify released data integrity, Python syntax, and public-file hygiene."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRS = {'analysis', 'evaluation', 'configs', 'data', 'docs', 'examples', 'scripts', 'tests', 'licenses'}
PUBLIC_ROOT_FILES = {'README.md', 'LICENSE', 'CITATION.cff', 'THIRD_PARTY.md',
                     'requirements.txt', 'requirements-features.txt', '.gitignore', 'SOURCE_MANIFEST.json'}
PUBLIC_SUFFIXES = {'.py', '.json', '.csv', '.md', '.txt'}


def public_files():
    paths = [ROOT / name for name in sorted(PUBLIC_ROOT_FILES)]
    for folder in sorted(PUBLIC_DIRS):
        paths.extend(p for p in sorted((ROOT / folder).rglob('*'))
                     if p.is_file() and p.suffix in PUBLIC_SUFFIXES and '__pycache__' not in p.parts)
    return paths


def verify():
    paths = public_files()
    for path in paths:
        if path.is_symlink():
            raise ValueError(f'Release must not contain symlinks: {path.name}')
        text = path.read_text(encoding='utf-8-sig')
        # Inspect payloads without embedding private host names in the scanner itself.
        forbidden_roots = ['/' + name + '/' for name in ['Users', 'mnt', 'home']]
        if any(root in text for root in forbidden_roots):
            raise ValueError(f'Private absolute path in {path.relative_to(ROOT)}')
        if re.search(r'(?:\d{1,3}\.){3}\d{1,3}', text):
            raise ValueError(f'IP address in {path.relative_to(ROOT)}')
        if re.search(r'(?:sk|hf)[_-][A-Za-z0-9]{24,}', text):
            raise ValueError(f'Possible access token in {path.relative_to(ROOT)}')
        if path.suffix == '.py':
            tree = ast.parse(text, filename=str(path.relative_to(ROOT)))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in {'unlink', 'rmtree', 'rmdir'}:
                        raise ValueError(f'Deletion operation in {path.relative_to(ROOT)}:{node.lineno}')
                    if node.func.attr == 'remove' and isinstance(node.func.value, ast.Name) and node.func.value.id == 'os':
                        raise ValueError(f'Deletion operation in {path.relative_to(ROOT)}:{node.lineno}')
    manifest = json.loads((ROOT / 'data/SHA256SUMS.json').read_text())
    for relative, expected in manifest.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Data checksum mismatch: {relative}')
    data_files = {p.relative_to(ROOT).as_posix() for p in paths
                  if 'data' == p.relative_to(ROOT).parts[0] and p.name != 'SHA256SUMS.json'}
    if data_files != set(manifest):
        raise ValueError('Data checksum manifest does not cover exactly the released data files')
    report = {'public_files': len(paths), 'verified_data_files': len(manifest),
              'python_sources': sum(p.suffix == '.py' for p in paths),
              'deletion_calls': 0, 'private_path_or_token_findings': 0}
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    verify()
