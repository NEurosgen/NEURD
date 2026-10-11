"""Shared completion checks and provenance for raw and branch batch runs.

The module uses only the standard library at import time. Geometry helpers and
raw runtime inspection import their numerical dependencies when called. Loading
this file from a pinned worker does not activate the NEURD package initializer.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def dependency_signature():
    versions = {}
    for name in ('numpy','scipy','trimesh','networkx','meshparty','mesh-processing-tools','datasci-stdlib-tools'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    files = {}
    for name in ('cgal_Segmentation_Module','mesh_tools'):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            raise RuntimeError(f'Missing required dependency: {name}')
        path = Path(spec.origin)
        files[str(path)] = sha(path)
        if name=='mesh_tools':
            for basename in ('mesh_utils.py','trimesh_utils.py','skeleton_utils.py','meshparty_skeletonize.py'):
                module = path.parent/basename
                if module.is_file():
                    files[str(module)] = sha(module)
    return dict(python=sys.version,executable=sys.executable,versions=versions,source_files=files)


def save(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def separate_roots(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Input and output roots must be separate, non-nested directories')
    return source, output


def triangle_keys(mesh):
    """Exact triangle geometry, independent of face/vertex order and winding."""
    import numpy as np
    rows = np.asarray(mesh.triangles,dtype=np.float64).copy().view(np.dtype((np.void,24))).reshape(-1,3)
    rows.sort(axis=1)
    return np.sort(rows.copy().view(np.dtype((np.void,72))).reshape(-1))


def complete_valid(dest, run_hash, source_inputs):
    """Verify completion hashes and inventory; scientific QC is separate."""
    result_path = Path(dest) / 'result.json'
    if not result_path.is_file():
        return False
    record = json.loads(result_path.read_text())
    if not isinstance(record, dict):
        raise ValueError(f'Malformed completion record: {result_path}')
    if record.get('status') != 'complete' or record.get('run_hash') != run_hash or record.get('inputs') != source_inputs:
        raise ValueError(f'Existing result has changed inputs or settings: {dest}')
    outputs = record.get('outputs_sha256')
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError(f'Completion record requires a nonempty output hash map: {result_path}')
    excluded = {'run.log', 'result.json'}
    for relative, expected in outputs.items():
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or '..' in Path(relative).parts or Path(relative).name in excluded
                or not isinstance(expected, str) or len(expected) != 64
                or any(character not in '0123456789abcdef' for character in expected)):
            raise ValueError(f'Malformed completed output hash entry: {result_path}')
        path = Path(dest)/relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Completed output is missing or changed: {path}')
    actual = {str(path.relative_to(dest)) for path in Path(dest).rglob('*')
              if (path.is_file() or path.is_symlink()) and path.name not in excluded}
    if set(outputs) != actual:
        raise ValueError(f'Completed output hash inventory is incomplete or changed: {dest}')
    return True


def raw_run_signature(config, decimation_ratio=0.0, max_somas=None, *, source_root=None):
    """Fingerprint configuration, loaded code, native providers and behavior flags.

    ``source_root`` selects an equivalent source snapshot for verification; by
    default it is this module's package location. Every package Python module,
    including this helper, remains in the signature. Repository runner scripts
    are included when present; installed packages do not require those scripts.
    """
    from neurd.run_config import runtime_environment

    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[1]
    code = [root / name for name in ("process_all_neurons.py", "process_all_neurons_parallel.py")
            if (root / name).is_file()]
    code.extend(sorted((root / "neurd").glob("*.py")))
    dependencies = dependency_signature()
    versions = dependencies.setdefault("versions", {})
    for name in ("open3d", "numba", "pymeshfix", "pykdtree", "embreex", "pyembree", "rtree"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    # Raw decomposition uses a separate native skeletonizer; H01 branch
    # resegmentation may retain stored skeletons and need only segmentation.
    skeletonizer = importlib.util.find_spec("calcification_param_Module")
    dependencies["skeletonizer"] = (
        dict(path=skeletonizer.origin, sha256=sha(skeletonizer.origin))
        if skeletonizer is not None and skeletonizer.origin else None
    )
    return dict(
        schema_version=1, config=config.as_dict(), decimation_ratio=decimation_ratio,
        max_somas=max_somas, code_sha256={str(path.relative_to(root)): sha(path) for path in code},
        dependencies=dependencies, runtime_environment=runtime_environment(config),
    )
