"""Compare legacy and owned mesh loading in fresh, interleaved processes.

Both routes import NEURD and mesh_tools before timing. Kernel lifetime peak RSS
includes those shared imports; it is not an allocation-delta measurement.

    python tests/tools/benchmark_mesh_load.py --mesh /path/neuron.off \
        --repeat 3 --out /tmp/mesh-load-ab.json

Use --legacy-only to record a baseline before neurd.mesh_io is available.
Input files are read without copying, conversion, caching or modification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')
CHILD = r'''
import hashlib, inspect, json, resource, sys, time
import numpy as np
import trimesh
import neurd
from mesh_tools import trimesh_utils as tu
from trimesh.exchange import off

mode, path = sys.argv[1:3]
if mode == 'owned':
    from neurd.mesh_io import load_mesh_no_processing as load
else:
    load = tu.load_mesh_no_processing

used_legacy_loader = False
original_legacy_loader = trimesh.load_mesh
def tracked_legacy_loader(*args, **kwargs):
    global used_legacy_loader
    used_legacy_loader = True
    return original_legacy_loader(*args, **kwargs)
trimesh.load_mesh = tracked_legacy_loader

def peak_rss_kib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024 if sys.platform == 'darwin' else value

def array_signature(array):
    array = np.asarray(array)
    digest = hashlib.sha256()
    if array.size and array.flags.c_contiguous:
        digest.update(memoryview(array).cast('B'))
    else:
        for row in array:
            digest.update(np.ascontiguousarray(row).tobytes())
    return dict(shape=list(array.shape), dtype=array.dtype.str, sha256=digest.hexdigest())

before = peak_rss_kib()
started = time.perf_counter()
mesh = load(path)
seconds = time.perf_counter() - started
after = peak_rss_kib()
sources = {}
for function in (load, off.load_off):
    filename = inspect.getsourcefile(function)
    with open(filename, 'rb') as stream:
        sources[filename] = hashlib.sha256(stream.read()).hexdigest()
receipt = dict(mode=mode, load_seconds=seconds,
    used_legacy_loader=used_legacy_loader,
    peak_rss_before_load_kib=before, peak_rss_kib=after,
    arrays=dict(vertices=array_signature(mesh.vertices), faces=array_signature(mesh.faces)),
    metadata=mesh.metadata, visual=dict(kind=mesh.visual.kind, defined=bool(mesh.visual.defined)),
    loader=dict(module=load.__module__, name=load.__name__, source_sha256=sources),
    versions=dict(numpy=np.__version__, trimesh=trimesh.__version__),
    import_patches=dict(neurd.PATCH_STATUS))
print('NEURD_MESH_LOAD_RESULT ' + json.dumps(receipt, sort_keys=True, allow_nan=False))
'''


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_metadata(path):
    stat = Path(path).stat()
    return dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns, device=stat.st_dev, inode=stat.st_ino)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--legacy-only', action='store_true')
    parser.add_argument('--timeout', type=int, default=600, help='positive seconds per child')
    args = parser.parse_args(argv)
    if min(args.repeat,args.timeout) <= 0:
        parser.error('repeat and timeout must be positive')
    mesh, output = args.mesh.expanduser().resolve(), args.out.expanduser().resolve()
    if not mesh.is_file():
        parser.error(f'Missing input mesh: {mesh}')
    if mesh == output or (output.exists() and output.samefile(mesh)):
        parser.error('Output receipt must be separate from the input mesh')
    original_hash, original_stat = sha(mesh), file_metadata(mesh)
    code = [Path(__file__).resolve(), ROOT/'neurd/__init__.py']
    owned_source = ROOT/'neurd/mesh_io.py'
    if not args.legacy_only and not owned_source.is_file():
        parser.error('Owned loader is unavailable; use --legacy-only to record the baseline')
    if owned_source.is_file():
        code.append(owned_source)
    code_hashes = {str(path):sha(path) for path in code}
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for name in THREAD_VARS:
        env.setdefault(name,'1')
    environment_names = [*THREAD_VARS, 'PYTHONHASHSEED', 'NEURD_LEGACY_MESH_LOADER',
                         'NEURD_LEGACY_VERTEX_COMPONENTS', 'NEURD_LEGACY_MATCHING_VERTICES',
                         'NEURD_MESHLAB_DECIMATE', 'NEURD_ENABLE_COMPRESSED_PICKLE']
    rows = []
    for trial in range(args.repeat):
        order = ('legacy',) if args.legacy_only else (
            ('legacy','owned') if trial % 2 == 0 else ('owned','legacy'))
        for mode in order:
            with tempfile.TemporaryDirectory(prefix='neurd_mesh_load_bench_') as directory:
                started = time.perf_counter()
                process = subprocess.run([sys.executable,'-c',CHILD,mode,str(mesh)],
                    env=env,cwd=directory,text=True,capture_output=True,timeout=args.timeout,check=True)
                wall = time.perf_counter()-started
                marker = 'NEURD_MESH_LOAD_RESULT '
                line = next(line for line in process.stdout.splitlines() if line.startswith(marker))
                row = json.loads(line.removeprefix(marker))
                row.update(trial=trial,wall_seconds=wall)
                rows.append(row)
                print(f"{mode} trial={trial} load_seconds={row['load_seconds']:.4f} "
                      f"peak_rss_kib={row['peak_rss_kib']:.0f} "
                      f"used_legacy_loader={row['used_legacy_loader']}",flush=True)
    exact = all((row['arrays'],row['metadata'],row['visual']) ==
                (rows[0]['arrays'],rows[0]['metadata'],rows[0]['visual']) for row in rows)
    source_unchanged = sha(mesh) == original_hash and file_metadata(mesh) == original_stat
    code_unchanged = all(path.is_file() and sha(path) == code_hashes[str(path)] for path in code)
    modes = ('legacy',) if args.legacy_only else ('legacy','owned')
    receipt = dict(input=dict(path=str(mesh),sha256=original_hash,**original_stat),
        python=sys.version,executable=sys.executable,platform=platform.platform(),cpu=platform.processor(),
        environment={name:env.get(name) for name in environment_names},code_sha256=code_hashes,
        legacy_only=args.legacy_only,exact_arrays_and_metadata=exact,input_unchanged=source_unchanged,
        code_unchanged=code_unchanged,trials=rows,
        summary={mode:dict(median_load_seconds=statistics.median(
            row['load_seconds'] for row in rows if row['mode']==mode),
            median_wall_seconds=statistics.median(row['wall_seconds'] for row in rows if row['mode']==mode),
            peak_rss_kib=max(row['peak_rss_kib'] for row in rows if row['mode']==mode),
            legacy_loader_trials=sum(row['used_legacy_loader'] for row in rows if row['mode']==mode)) for mode in modes})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    if not exact or not source_unchanged or not code_unchanged:
        raise SystemExit('FAIL: arrays/metadata differed or input/code changed; see receipt')
    print(json.dumps(receipt['summary'],indent=2))


if __name__ == '__main__':
    main()
