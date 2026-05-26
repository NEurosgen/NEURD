# Running tests inside Docker

The NEURD dependency stack (`datasci_tools`, `mesh_tools`, `meshparty`,
`trimesh==3.22.3`, system libs for embree/CGAL/meshlab) is fragile to install
locally — pinned to `numpy<2`, `open3d==0.11.2`, etc. The supported way is the
Docker image already defined in this folder.

## One-time build

```bash
cd docker/
docker compose build test     # ~10–30 min on first run; cached afterwards
```

The `test` service reuses the same image as `notebook` (`celiib/neurd:v1`),
so building it warms the cache for both.

## Run all unit tests

```bash
cd docker/
docker compose run --rm test
```

This mounts the repo at `/NEURD`, runs `pip install -e .` (so any local
edits in `neurd/` take effect without rebuilding the image), then
`pytest tests/unit/ -v`.

## Run a specific test file / pattern

```bash
docker compose run --rm test bash -c "pip install -q -e . && pytest tests/unit/leaves/test_volume_utils.py -v"

# or just the failing case:
docker compose run --rm test bash -c "pip install -q -e . && pytest tests/unit/leaves -k cdist -v"
```

## Run integration tests

```bash
docker compose run --rm test bash -c "pip install -q -e . && pytest tests/integration/ -v"
```

Integration tests need `tests/fixtures/864691135510518224*.{off,csv}` — those
are tracked in the repo (see `.gitignore` exclusion).

## Interactive shell inside the test container

```bash
docker compose run --rm test bash
# then inside:
pip install -e .
pytest tests/unit/leaves/ -v
python                # interpreter with neurd importable
```

## Notes

- The `notebook` service still uses `env_file: .env`, but the file is now
  marked `required: false`, so the absence of `docker/.env` no longer breaks
  `docker compose run --rm test`.
- The `test` service does **not** install NEURD at image build time — only at
  container start (`pip install -e .` against the mount). This lets you edit
  `neurd/` on the host and re-run tests without rebuilding.
- If you want a faster test cycle, run an interactive shell (see above) and
  invoke `pytest` repeatedly inside it.
