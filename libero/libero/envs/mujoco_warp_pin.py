"""Fail before GPU initialization when the required fork is not installed."""
import hashlib
from importlib import metadata
import json

REPOSITORY = 'https://github.com/ShahRutav/mujoco_warp.git'
BRANCH = 'fix/elliptic-regularization-floor'
COMMIT = 'cac9e68c6369a5fb9970a12118aec77c27154ffc'
CONSTRAINT_SHA256 = 'ec1417159ffc08bc8aa401b5be121104f5aad3ca877eda4a3371f781f3079f57'


def verify():
    dist = metadata.distribution('mujoco-warp')
    origin = json.loads(dist.read_text('direct_url.json') or '{}')
    source = dist.locate_file('mujoco_warp/_src/constraint.py')
    if (origin.get('url') != REPOSITORY
            or origin.get('vcs_info', {}).get('commit_id') != COMMIT
            or hashlib.sha256(source.read_bytes()).hexdigest() != CONSTRAINT_SHA256):
        raise RuntimeError(
            f'LIBERO GPU runs require {REPOSITORY}@{COMMIT} ({BRANCH}). '
            'Install requirements-mjlab.txt with the documented dependency overrides.')
    return origin
