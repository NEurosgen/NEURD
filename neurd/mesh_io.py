"""Read triangular ASCII OFF meshes without retaining the complete text file.

The fast path preserves float64 vertices and int64 triangle indices, including
unused vertices, duplicate faces, and winding. Other formats and OFF variants
are delegated to ``trimesh.load_mesh(..., process=False)``. There is no geometry
cleaning, persistent cache, or change to the source file.
"""

import os
from pathlib import Path
import re
import warnings

import numpy as np
import trimesh


# These limits bound temporary text and conversion arrays independently of mesh
# size. Unusually long records use the existing loader rather than redefining
# the accepted OFF variants.
_CHUNK_BYTES = 1024 * 1024
_MAX_LINE_BYTES = 64 * 1024
_INT64_MAX = b"9223372036854775807"
_INT64_MIN_ABS = b"9223372036854775808"
_SPECIAL_RECORD = re.compile(rb"[\x00\x0b\x0c\x1c-\x1e]|\r(?!\n?$)")
# At most 18 unsigned decimal digits always fit in int64. Check the usual
# triangle grammar in C; signed/long/attributed records retain slow validation.
_PLAIN_TRIANGLE = re.compile(rb"3[ \t]+[0-9]{1,18}[ \t]+[0-9]{1,18}[ \t]+[0-9]{1,18}")


class _UseLegacy(Exception):
    """The input is outside the deliberately small optimized OFF grammar."""


def _records(stream):
    """Yield nonempty ASCII records, stripping the same # comments as trimesh."""
    records_seen = 0
    comment_seen = False
    while True:
        line = stream.readline(_MAX_LINE_BYTES + 1)
        if not line:
            return
        # str.splitlines in the existing parser treats these as record
        # boundaries, while binary readline only recognizes LF.
        if len(line) > _MAX_LINE_BYTES or not line.isascii() or _SPECIAL_RECORD.search(line):
            raise _UseLegacy
        if not comment_seen and b"#" in line:
            # Installed trimesh.comment_strip duplicates the prefix after its
            # first newline when the first comment appears later in the file.
            # Preserve that legacy behavior by delegating these layouts.
            if records_seen:
                raise _UseLegacy
            comment_seen = True
        if b"#" in line:
            line = line.split(b"#", 1)[0]
        line = line.strip()
        if line:
            records_seen += 1
            yield line


def _integer_token(token):
    """Validate before fromstring, which can silently saturate int64 overflow."""
    if len(token) <= 18 and token.isdigit():
        return
    negative = token.startswith(b"-")
    digits = token[1:] if token.startswith((b"-", b"+")) else token
    if not digits or not digits.isdigit():
        raise ValueError("OFF integer fields must contain decimal integers")
    digits = digits.lstrip(b"0") or b"0"
    limit = _INT64_MIN_ABS if negative else _INT64_MAX
    if len(digits) > len(limit) or (len(digits) == len(limit) and digits > limit):
        raise ValueError("OFF integer field exceeds int64 range")


def _next_record(records, section):
    try:
        return next(records)
    except StopIteration:
        raise ValueError(f"Truncated OFF {section}") from None


def _convert_chunk(text, count, width, dtype, section):
    try:
        # Older numpy warns and returns partial arrays for unmatched text;
        # newer versions raise ValueError. Both must reject the whole chunk.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            values = np.fromstring(bytes(text), dtype=dtype, sep=" ")
    except (ValueError, DeprecationWarning) as exc:
        raise ValueError(f"Malformed OFF {section}: invalid numeric token") from exc
    if values.size != count * width:
        raise ValueError(f"Malformed OFF {section}: incorrect numeric field count")
    values = values.reshape(count, width)
    if section == "vertices" and not np.isfinite(values).all():
        raise ValueError("OFF vertices must be finite float64 values")
    return values


def _read_array(records, count, *, vertices, vertex_count=0):
    section = "vertices" if vertices else "faces"
    width = 3 if vertices else 4
    dtype = np.float64 if vertices else np.int64
    # trimesh's OFF loader gives empty faces shape (0,), not (0, 3).
    shape = (count, 3) if vertices or count else (0,)
    output = np.empty(shape, dtype=dtype)
    chunk = bytearray()
    chunk_count = 0
    output_start = 0

    def flush():
        nonlocal output_start, chunk_count
        values = _convert_chunk(chunk, chunk_count, width, dtype, section)
        if not vertices:
            values = values[:, 1:]
            if values.size and (values.min() < 0 or values.max() >= vertex_count):
                raise ValueError("OFF face index is outside the vertex array")
        output[output_start:output_start + chunk_count] = values
        output_start += chunk_count
        chunk.clear()
        chunk_count = 0

    for _ in range(count):
        line = _next_record(records, section)
        if not vertices and _PLAIN_TRIANGLE.fullmatch(line):
            pass
        else:
            tokens = line.split()
            if vertices:
                if len(tokens) < 3:
                    raise ValueError("OFF vertex record needs three coordinates")
                if len(tokens) > 3:
                    raise _UseLegacy  # Colors, normals, or another OFF variant.
            else:
                if tokens[0] != b"3":
                    _integer_token(tokens[0])
                    if int(tokens[0]) != 3:
                        raise _UseLegacy  # Preserve legacy polygon triangulation.
                if len(tokens) < 4:
                    raise ValueError("OFF triangle record needs three indices")
                if len(tokens) > 4:
                    raise _UseLegacy  # Face colors or attributes.
                for token in tokens[1:]:
                    _integer_token(token)
        chunk.extend(line)
        chunk.append(10)
        chunk_count += 1
        if len(chunk) >= _CHUNK_BYTES:
            flush()
    if chunk_count:
        flush()
    return output


def _load_ascii_off(path):
    with path.open("rb") as stream:
        records = _records(stream)
        first = _next_record(records, "header").split()
        if first[0] != b"OFF":
            raise _UseLegacy
        counts = first[1:] or _next_record(records, "counts").split()
        if counts[0].upper() == b"BINARY":
            raise _UseLegacy
        if len(counts) < 2:
            raise ValueError("OFF header needs vertex and face counts")
        for token in counts:
            _integer_token(token)
        vertex_count, face_count = map(int, counts[:2])
        if vertex_count < 0 or face_count < 0:
            raise ValueError("OFF vertex and face counts must be nonnegative")
        # Even minimal records need '0 0 0' / '3 0 0 0'. Check before
        # allocating arrays so a tiny corrupt file cannot request huge memory.
        remaining_bytes = os.fstat(stream.fileno()).st_size - stream.tell()
        if vertex_count * 5 + face_count * 7 > remaining_bytes:
            # A short binary body is an unsupported variant, not malformed
            # ASCII. Inspect one bounded record before reporting truncation.
            next(records, None)
            raise ValueError("Truncated OFF data for the declared counts")
        vertices = _read_array(records, vertex_count, vertices=True)
        faces = _read_array(records, face_count, vertices=False,
                            vertex_count=vertex_count)
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        metadata={"name": path.name, "node": path.name},
    )


def load_mesh(path, *, force_legacy=False):
    """Load a local mesh with no processing and bounded ASCII OFF temporaries.

    Plain ASCII triangular ``.off`` files use the optimized reader. Unsupported
    variants, other extensions, and non-path inputs use trimesh's existing
    loader. Comments before the header or in its first record are supported;
    later first comments use the legacy loader to preserve its prefix handling.
    Invalid counts, nonfinite coordinates, and invalid triangle indices
    in the fast grammar raise ``ValueError``. Callers intentionally repairing
    invalid source indices must select ``force_legacy=True``. The environment
    variable ``NEURD_LEGACY_MESH_LOADER=1`` also selects the existing loader.

    Unlike mesh_tools' historical wrapper, this API does not append an extension
    to a filename. Pass the actual file path.
    """
    legacy = force_legacy or os.environ.get("NEURD_LEGACY_MESH_LOADER") == "1"
    if not legacy and isinstance(path, (str, os.PathLike)):
        local_path = Path(path)
        if local_path.suffix.lower() == ".off":
            try:
                return _load_ascii_off(local_path)
            except _UseLegacy:
                pass
    return trimesh.load_mesh(path, process=False)


def load_mesh_no_processing(path, *, force_legacy=False):
    """Alias for ``load_mesh`` for callers describing the processing contract."""
    return load_mesh(path, force_legacy=force_legacy)
