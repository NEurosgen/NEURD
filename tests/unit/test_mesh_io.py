"""Exact no-processing OFF compatibility and bounded temporary-buffer checks."""

from pathlib import Path
import warnings
import weakref

import numpy as np
import pytest
import trimesh

from neurd import mesh_io


TRIANGLE = "3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"


@pytest.fixture(autouse=True)
def no_legacy_environment(monkeypatch):
    monkeypatch.delenv("NEURD_LEGACY_MESH_LOADER", raising=False)


def _file(tmp_path, contents, name="test.off"):
    path = tmp_path / name
    if isinstance(contents, bytes):
        path.write_bytes(contents)
    else:
        path.write_text(contents, encoding="utf-8")
    return path


def _assert_exact(actual, expected):
    assert actual.vertices.dtype == expected.vertices.dtype == np.float64
    assert actual.faces.dtype == expected.faces.dtype == np.int64
    assert actual.vertices.shape == expected.vertices.shape
    assert actual.faces.shape == expected.faces.shape
    # Byte comparison also distinguishes signed zeros and adjacent float64s.
    assert actual.vertices.tobytes() == expected.vertices.tobytes()
    assert actual.faces.tobytes() == expected.faces.tobytes()
    np.testing.assert_equal(actual.metadata, expected.metadata)
    assert actual.visual.kind == expected.visual.kind
    assert actual.visual.defined == expected.visual.defined
    assert actual._cache.cache == expected._cache.cache == {}
    assert actual.source == expected.source


@pytest.mark.parametrize("header", ["OFF\n", "OFF ", "\n# header comment\nOFF\t"])
def test_order_duplicates_unused_vertices_and_comments(tmp_path, header):
    path = _file(tmp_path, "# file comment\n" + header + """6 4 0 # counts

0.12345678901234567 -0.0 1.2345678901234567e-99
1 0 0 # vertex
0 1 0
0 0 1
1 0 0
99.99999999999999 3 4 # unused vertex
# winding and duplicates are meaningful
3 2 1 0
3 0 1 2
3 0 1 2
3 0 4 3
ignored trailing data
""")
    before = path.read_bytes()
    expected = trimesh.load_mesh(path, process=False)
    actual = mesh_io.load_mesh(path)
    _assert_exact(actual, expected)
    assert len(actual.vertices) == 6
    assert len(actual.faces) == 4
    assert path.read_bytes() == before


def test_float64_decimal_conversion_exact_at_extremes(tmp_path):
    rng = np.random.default_rng(341)
    values = rng.integers(0, np.iinfo(np.uint64).max, size=513,
                          dtype=np.uint64).view(np.float64)
    values = values[np.isfinite(values)]
    values = np.concatenate(([0.0, -0.0, np.nextafter(0.0, 1.0),
                              -np.nextafter(0.0, 1.0),
                              np.finfo(np.float64).max,
                              np.finfo(np.float64).tiny], values))
    values = values[:len(values) // 3 * 3].reshape(-1, 3)
    text = "OFF\n{} 1 0\n".format(len(values))
    text += "\n".join(" ".join(format(x, ".17g") for x in row) for row in values)
    path = _file(tmp_path, text + "\n3 0 1 2\n")
    _assert_exact(mesh_io.load_mesh(path), trimesh.load_mesh(path, process=False))


@pytest.mark.parametrize("body", ["0 0 0\n", "1 0 0\n-0 1 2\n"])
def test_empty_faces_match_legacy_shape(tmp_path, body):
    path = _file(tmp_path, "OFF\n" + body)
    _assert_exact(mesh_io.load_mesh(path), trimesh.load_mesh(path, process=False))


def test_ascii_whitespace_and_crlf(tmp_path):
    path = _file(tmp_path, b"OFF # file\r\n3\t1\t0\r\n0\t0 0\r\n1\t0 0\r\n"
                          b"0 1\t0\r\n+3 +0 01 +2 # face\r\n")
    _assert_exact(mesh_io.load_mesh(path), trimesh.load_mesh(path, process=False))


@pytest.mark.parametrize("face", [
    "3 0 1 2",
    "3 000000000000000000 000000000000000001 000000000000000002",
    "3 0000000000000000000 0000000000000000001 0000000000000000002",
    "03 +0 +1 +2", "+3 -0 1 2", "3 \t0\t 1 \t2",
])
def test_integer_spelling_and_digit_boundaries_are_exact(tmp_path, face):
    path = _file(tmp_path, "OFF\n" + TRIANGLE.replace("3 0 1 2", face))
    _assert_exact(mesh_io.load_mesh(path), trimesh.load_mesh(path, process=False))


@pytest.mark.parametrize("body", [
    "COFF\n3 1 0\n0 0 0 255 0 0 255\n1 0 0 0 255 0 255\n"
    "0 1 0 0 0 255 255\n3 0 1 2\n",
    "NOFF\n3 1 0\n0 0 0 0 0 1\n1 0 0 0 0 1\n0 1 0 0 0 1\n3 0 1 2\n",
    "OFF\n3 1 0\n0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 255\n3 0 1 2\n",
    "OFF\n4 1 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n",
    "OFF\n5 1 0\n0 0 0\n1 0 0\n2 1 0\n1 2 0\n0 1 0\n5 0 1 2 3 4\n",
    "OFF\n" + TRIANGLE.replace("3 0 1 2", "3 0 1 2 255 0 0"),
    "# комментарий\nOFF\n" + TRIANGLE,
    "OFF\n" + TRIANGLE.replace("1 0 0", "1\u20020 0"),
    "OFF\n" + TRIANGLE.replace("\n", "\r"),
])
def test_unsupported_variants_delegate_and_remain_exact(tmp_path, monkeypatch, body):
    path = _file(tmp_path, body)
    legacy = trimesh.load_mesh
    expected = legacy(path, process=False)
    calls = []

    def recorded(*args, **kwargs):
        calls.append((args, kwargs))
        return legacy(*args, **kwargs)

    monkeypatch.setattr(mesh_io.trimesh, "load_mesh", recorded)
    _assert_exact(mesh_io.load_mesh(path), expected)
    assert calls == [((path,), {"process": False})]


def test_non_off_file_uses_existing_loader(tmp_path, monkeypatch):
    path = tmp_path / "triangle.ply"
    trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    faces=[[0, 1, 2]], process=False).export(path)
    expected = trimesh.load_mesh(path, process=False)
    monkeypatch.setattr(mesh_io, "_load_ascii_off",
                        lambda path: pytest.fail("non-OFF entered fast parser"))
    _assert_exact(mesh_io.load_mesh(path), expected)


@pytest.mark.parametrize("payload", [
    b"OFF BINARY\n\x00\x01\x02\xff",
    b"OFF\nBINARY\n\x00\x01\x02\xff",
    b"OFF\n3 1 0\n\x00\x01\x02\xff",
])
def test_binary_off_delegates(tmp_path, monkeypatch, payload):
    path = _file(tmp_path, payload)
    sentinel = object()
    calls = []

    def legacy(path, **kwargs):
        calls.append((path, kwargs))
        return sentinel

    monkeypatch.setattr(mesh_io.trimesh, "load_mesh", legacy)
    assert mesh_io.load_mesh(path) is sentinel
    assert calls == [(path, {"process": False})]


@pytest.mark.parametrize("selection", ["keyword", "environment"])
def test_legacy_switch_preserves_repairable_invalid_indices(tmp_path, monkeypatch, selection):
    path = _file(tmp_path, "OFF\n" + TRIANGLE.replace("3 0 1 2", "3 5 6 7"))
    expected = trimesh.load_mesh(path, process=False)
    with pytest.raises(ValueError, match="outside the vertex"):
        mesh_io.load_mesh(path)
    if selection == "environment":
        monkeypatch.setenv("NEURD_LEGACY_MESH_LOADER", "1")
    actual = mesh_io.load_mesh(path, force_legacy=selection == "keyword")
    _assert_exact(actual, expected)


def test_no_processing_alias_uses_actual_path(tmp_path):
    path = _file(tmp_path, "OFF\n" + TRIANGLE)
    _assert_exact(mesh_io.load_mesh_no_processing(str(path)),
                  trimesh.load_mesh(path, process=False))


def test_non_path_input_delegates_without_coercion(monkeypatch):
    input_object = object()
    output_object = object()

    def legacy(obj, **kwargs):
        assert obj is input_object
        assert kwargs == {"process": False}
        return output_object

    monkeypatch.setattr(mesh_io.trimesh, "load_mesh", legacy)
    assert mesh_io.load_mesh(input_object) is output_object


@pytest.mark.parametrize("body", [
    "", "# empty\n", "OFF\n", "OFF\n3\n", "OFF\n-1 0 0\n",
    "OFF\n9999999999999999999 0 0\n",
    "OFF\n3 1 nope\n", "OFF\n3 1 0\n0 0 0\n",
    "OFF\n" + TRIANGLE.replace("0 1 0", "0 1"),
    "OFF\n" + TRIANGLE.replace("3 0 1 2", "3 0 1"),
    "OFF\n" + TRIANGLE.replace("0 1 0", "0 1 0garbage"),
    "OFF\n" + TRIANGLE.replace("0 1 0", "0 nan 0"),
    "OFF\n" + TRIANGLE.replace("0 1 0", "0 inf 0"),
    "OFF\n" + TRIANGLE.replace("0 1 0", "0 1e999 0"),
    "OFF\n" + TRIANGLE.replace("3 0 1 2", "3 -1 1 2"),
    "OFF\n" + TRIANGLE.replace("3 0 1 2", "3 0 1 3"),
    "OFF\n" + TRIANGLE.replace("3 0 1 2", "3 0 1 2.0"),
    "OFF\n" + TRIANGLE.replace("3 0 1 2", "3 0 1 2garbage"),
    "OFF\n" + TRIANGLE.replace("3 0 1 2", "3 0 1 +-2"),
    "OFF\n" + TRIANGLE.replace("3 0 1 2", "3 0 1 9223372036854775808"),
    "OFF\n" + TRIANGLE.replace("3 0 1 2", "3 0 1 -9223372036854775809"),
])
def test_malformed_fast_inputs_fail_without_legacy_or_source_changes(tmp_path, monkeypatch, body):
    path = _file(tmp_path, body)
    before = path.read_bytes()
    monkeypatch.setattr(mesh_io.trimesh, "load_mesh",
                        lambda *args, **kwargs: pytest.fail("malformed fast input delegated"))
    with pytest.raises(ValueError):
        mesh_io.load_mesh(path)
    assert path.read_bytes() == before


def test_impossible_counts_rejected_before_allocation(tmp_path, monkeypatch):
    path = _file(tmp_path, "OFF\n1000000000000 1000000000000 0\n")
    monkeypatch.setattr(mesh_io.np, "empty",
                        lambda *args, **kwargs: pytest.fail("declared counts allocated"))
    with pytest.raises(ValueError, match="declared counts"):
        mesh_io.load_mesh(path)


def test_old_numpy_partial_parse_warning_is_an_error(tmp_path, monkeypatch):
    path = _file(tmp_path, "OFF\n" + TRIANGLE)

    def partial(*args, **kwargs):
        warnings.warn("unmatched data", DeprecationWarning)
        return np.asarray([0.0, 0.0, 0.0])

    monkeypatch.setattr(mesh_io.np, "fromstring", partial)
    with pytest.raises(ValueError, match="invalid numeric token"):
        mesh_io.load_mesh(path)


def test_chunked_reads_and_conversions_are_bounded(tmp_path, monkeypatch):
    count = 80
    text = f"OFF\n{count} {count} 0\n"
    text += "".join(f"{i}.12345678901234567 0 -0\n" for i in range(count))
    text += "".join(f"3 {i} {(i + 1) % count} {(i + 2) % count}\n" for i in range(count))
    path = _file(tmp_path, text)
    expected = trimesh.load_mesh(path, process=False)
    original_open = Path.open
    original_fromstring = np.fromstring
    read_sizes = []
    conversion_sizes = []
    monkeypatch.setattr(mesh_io, "_CHUNK_BYTES", 96)

    class Reader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def readline(self, size):
            assert 0 < size <= mesh_io._MAX_LINE_BYTES + 1
            read_sizes.append(size)
            return self.stream.readline(size)

        def read(self, *args):
            pytest.fail("fast parser attempted whole-file read")

        def fileno(self):
            return self.stream.fileno()

        def tell(self):
            return self.stream.tell()

    def bounded_open(target, *args, **kwargs):
        stream = original_open(target, *args, **kwargs)
        return Reader(stream) if target == path else stream

    def bounded_conversion(data, *args, **kwargs):
        conversion_sizes.append(len(data))
        return original_fromstring(data, *args, **kwargs)

    monkeypatch.setattr(Path, "open", bounded_open)
    monkeypatch.setattr(mesh_io.np, "fromstring", bounded_conversion)
    _assert_exact(mesh_io.load_mesh(path), expected)
    assert len(read_sizes) == 2 + 2 * count
    assert len(conversion_sizes) > 10
    assert max(conversion_sizes) <= 96 + max(map(len, text.splitlines())) + 1


def test_oversized_comment_record_delegates(tmp_path, monkeypatch):
    path = _file(tmp_path, "#" + "x" * (mesh_io._MAX_LINE_BYTES + 1)
                 + "\nOFF\n" + TRIANGLE)
    legacy = trimesh.load_mesh
    expected = legacy(path, process=False)
    calls = []

    def recorded(*args, **kwargs):
        calls.append(1)
        return legacy(*args, **kwargs)

    monkeypatch.setattr(mesh_io.trimesh, "load_mesh", recorded)
    _assert_exact(mesh_io.load_mesh(path), expected)
    assert calls == [1]


def test_late_first_comment_preserves_legacy_prefix_behavior(tmp_path, monkeypatch):
    # This installed legacy layout repeats the counts as the sole vertex. The
    # optimized loader must not silently replace that observable behavior.
    path = _file(tmp_path, "OFF\n1 0 0 # counts\n8 9 10\n")
    legacy = trimesh.load_mesh
    expected = legacy(path, process=False)
    np.testing.assert_array_equal(expected.vertices, [[1, 0, 0]])
    calls = []

    def recorded(*args, **kwargs):
        calls.append(1)
        return legacy(*args, **kwargs)

    monkeypatch.setattr(mesh_io.trimesh, "load_mesh", recorded)
    _assert_exact(mesh_io.load_mesh(path), expected)
    assert calls == [1]


@pytest.mark.parametrize("separator", ["\v", "\f", "\x1c", "\x1d", "\x1e"])
def test_legacy_extra_record_separators_delegate(tmp_path, monkeypatch, separator):
    path = _file(tmp_path, "OFF\n" + TRIANGLE.replace("1 0 0", f"1{separator}0 0"))
    sentinel = object()
    monkeypatch.setattr(mesh_io.trimesh, "load_mesh", lambda *args, **kwargs: sentinel)
    assert mesh_io.load_mesh(path) is sentinel


def test_symlink_metadata_uses_caller_name(tmp_path):
    source = _file(tmp_path, "OFF\n" + TRIANGLE, "real.off")
    alias = tmp_path / "alias.off"
    alias.symlink_to(source)
    actual = mesh_io.load_mesh(alias)
    expected = trimesh.load_mesh(alias, process=False)
    _assert_exact(actual, expected)
    assert actual.metadata == {"name": "alias.off", "node": "alias.off"}


def test_fallback_does_not_retain_preallocated_arrays(tmp_path, monkeypatch):
    path = _file(tmp_path, "OFF\n" + TRIANGLE)
    arrays = []
    sentinel = object()

    def read_array(*args, vertices, **kwargs):
        if not vertices:
            raise mesh_io._UseLegacy
        vertices = np.empty((3, 3), dtype=np.float64)
        arrays.append(weakref.ref(vertices))
        return vertices

    def legacy(*args, **kwargs):
        assert len(arrays) == 1
        assert arrays[0]() is None
        return sentinel

    monkeypatch.setattr(mesh_io, "_read_array", read_array)
    monkeypatch.setattr(mesh_io.trimesh, "load_mesh", legacy)
    assert mesh_io.load_mesh(path) is sentinel
