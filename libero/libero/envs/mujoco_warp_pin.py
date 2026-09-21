"""Fail before GPU initialization when the required fork is not installed."""
import hashlib
from importlib import metadata
import json

REPOSITORY = 'https://github.com/ShahRutav/mujoco_warp.git'
BRANCH = 'fix/elliptic-normalized-curvature'
COMMIT = '9df8bdb02acdb9e2661bcfa4a7fef265952d35d2'
CONSTRAINT_SHA256 = 'ec1417159ffc08bc8aa401b5be121104f5aad3ca877eda4a3371f781f3079f57'

SOLVER_SHA256 = '4b671cafb3a2f5a4e4608baa857d63d1c8a7ca57c255de8b02e8e793374c1211'

def verify():
    dist = metadata.distribution('mujoco-warp')
    origin = json.loads(dist.read_text('direct_url.json') or '{}')
    constraint = dist.locate_file('mujoco_warp/_src/constraint.py')
    solver = dist.locate_file('mujoco_warp/_src/solver.py')
    if (origin.get('url') != REPOSITORY
            or origin.get('vcs_info', {}).get('commit_id') != COMMIT
            or hashlib.sha256(constraint.read_bytes()).hexdigest() != CONSTRAINT_SHA256
            or hashlib.sha256(solver.read_bytes()).hexdigest() != SOLVER_SHA256):
        raise RuntimeError(
            f'LIBERO GPU runs require {REPOSITORY}@{COMMIT} ({BRANCH}). '
            'Install requirements-mjlab.txt with the documented dependency overrides.')
    return origin
