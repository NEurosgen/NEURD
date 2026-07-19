# extract_spine_descriptors.py
# -*- coding: utf-8 -*-
#
# Обходит папку с мешами шипиков (.off/.ply/...) и для КАЖДОГО меша сохраняет:
#   * <stem>.npy          — вектор дескрипторов (float64, np.save)
#   * <stem>_centroid.npy — центроид меша в АБСОЛЮТНЫХ координатах (3 float, нм)
# Имя .npy дескрипторов повторяет имя файла меша (stem), у центроида +"_centroid".
# Рядом один раз пишется feature_names.json — порядок колонок в векторе.
#
# Дескрипторы считаются движком NEURD на готовых (уже вырезанных) мешах шипиков:
#   базовые атрибуты + head/neck-сегментация (настоящий CGAL SDF).
# head/neck зовётся напрямую (Spine.calculate_head_neck), В ОБХОД атрибута
# head_mesh_splits, который падает IndexError на безголовых шипиках.
#
# Параллелизм: ProcessPoolExecutor. Каждый воркер обрабатывает один меш в
# СВОЁМ приватном CWD (tempfile.mkdtemp): CGAL пишет временные .off/.csv в "./"
# со случайным префиксом 10..1000, поэтому общий CWD у воркеров = гонки и
# коллизии имён. Приватный CWD на задачу это полностью изолирует.
#
# Пример:
#   python extract_spine_descriptors.py \
#       /home/eugen/Desktop/CodeWork/Projects/NEURD_modif/Spine_example/h01 \
#       /home/eugen/Desktop/CodeWork/Projects/NEURD_modif/Spine_desc/h01 \
#       --workers 8
#   # рекурсивно по обоим датасетам, с зеркалированием подпапок:
#   python extract_spine_descriptors.py \
#       /home/eugen/Desktop/CodeWork/Projects/NEURD_modif/Spine_example \
#       /home/eugen/Desktop/CodeWork/Projects/NEURD_modif/Spine_desc \
#       --recursive --workers 8

import os

# Ограничиваем численные бэкенды ДО импорта numpy/neurd: иначе каждый из N
# воркеров поднимет свой пул BLAS-потоков и они подерутся за ядра.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import json
import time
import shutil
import tempfile
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# Порядок колонок вектора дескрипторов. ФИКСИРОВАН — не менять местами,
# иначе уже сохранённые .npy станут несопоставимы. Дописывать только в конец.
# head_* признаки = NaN, если у шипика не выделена голова (head_exist == 0).
FEATURE_NAMES = [
    # --- шипик целиком ---
    "n_faces",                    # число граней меша шипика
    "n_vertices",                 # число вершин
    "volume",                     # объём (нм^3), дыры заклеиваются (tu.mesh_volume)
    "area",                       # площадь поверхности (нм^2)
    "skeletal_length",            # длина поверхностного скелета шипика (нм)
    "spine_volume_to_spine_area", # отношение объём/площадь
    "bbox_side_min",              # ориентированный bbox: короткая сторона (нм)
    "bbox_side_mid",              # средняя сторона (нм)
    "bbox_side_max",              # длинная сторона (нм)
    # --- флаги головы ---
    "head_exist",                 # 1.0 если голова найдена, иначе 0.0
    "n_heads",                    # число голов (0.0 если головы нет)
    # --- голова (NaN, если head_exist == 0) ---
    "head_volume",                # объём головы (нм^3)
    "head_area",                  # площадь головы (нм^2)
    "head_volume_to_head_area",   # объём/площадь головы
    "head_skeletal_length",       # длина скелета головы (нм)
    "head_n_faces",               # граней в голове
    "head_width",                 # ширина головы (SDF-оценка, нм)
    "head_width_ray",             # ширина головы (ray-trace, 50-й перцентиль, нм)
    "head_width_ray_80_perc",     # ширина головы (ray-trace, 80-й перцентиль, нм)
    "head_sdf",                   # средний SDF головы
    "head_bbox_side_min",         # bbox головы: короткая сторона (нм)
    "head_bbox_side_mid",         # средняя сторона (нм)
    "head_bbox_side_max",         # длинная сторона (нм)
    # --- шейка (считается всегда) ---
    "neck_volume",                # объём шейки (нм^3)
    "neck_area",                  # площадь шейки (нм^2)
    "neck_volume_to_neck_area",   # объём/площадь шейки
    "neck_skeletal_length",       # длина шейки = длина скелета шейки (нм)
    "neck_n_faces",               # граней в шейке
    "neck_width",                 # ширина шейки (SDF-оценка, нм)
    "neck_width_ray",             # ширина шейки (ray-trace, 50-й перцентиль, нм)
    "neck_width_ray_80_perc",     # ширина шейки (ray-trace, 80-й перцентиль, нм)
    "neck_sdf",                   # средний SDF шейки
    "neck_bbox_side_min",         # bbox шейки: короткая сторона (нм)
    "neck_bbox_side_mid",         # средняя сторона (нм)
    "neck_bbox_side_max",         # длинная сторона (нм)
]

# Сентинелы NEURD для "головы нет" -> в .npy кладём NaN (чище для ML).
_NO_HEAD_SENTINEL = -1


def _f(x):
    """None/сентинел -> NaN, иначе float."""
    import math
    if x is None:
        return math.nan
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return math.nan
    return xf


def _compute_descriptors(path: str):
    """Считает (descriptor_vector, centroid) для одного меша.

    Импорты NEURD/mesh_tools — внутри функции и СТРОГО в этом порядке:
    neurd патчит datasci_tools.numpy_dep, иначе голый импорт mesh_tools
    падает на np.float_ (numpy>=2.0).
    """
    from neurd import spine_utils as spu      # ПЕРВЫМ
    from neurd import parameters              # noqa: F401 (режим ставится снаружи)
    from mesh_tools import trimesh_utils as tu
    import numpy as np

    mesh = tu.load_mesh_no_processing(path)

    # Центроид в абсолютных координатах — берём с СЫРОГО меша, до любой
    # обработки, которая могла бы его пересчитать/сместить.
    centroid = np.asarray(mesh.centroid, dtype=np.float64).reshape(3)

    sp = spu.Spine(mesh)
    # Базовые атрибуты (без head/neck): объём, скелет, bbox — устойчивы всегда.
    spu.calculate_spine_attributes(
        sp, branch_obj=None,
        calculate_coordinates=False,
        calculate_head_neck=False,
    )
    # head/neck напрямую — в обход head_mesh_splits (он падает на безголовых).
    sp = spu.calculate_spine_obj_mesh_skeleton_coordinates(spine_obj=sp, mesh=sp.mesh)
    sp.calculate_head_neck()

    nan = float("nan")

    def _bbox3(lengths):
        """3 стороны ориентированного bbox по возрастанию; NaN при отсутствии."""
        vals = [_f(x) for x in lengths]
        if len(vals) < 3 or any(v != v for v in vals):  # NaN среди значений
            return [nan, nan, nan]
        return sorted(vals)

    def _ratio(num, den):
        """num/den с защитой от нуля/NaN."""
        n_, d_ = _f(num), _f(den)
        if n_ != n_ or d_ != d_ or d_ == 0:
            return nan
        return n_ / d_

    # --- шипик целиком (всегда) ---
    bbox = _bbox3(sp.bbox_oriented_side_lengths)
    spine_vals = [
        _f(sp.n_faces),
        _f(sp.n_vertices),
        _f(sp.volume),
        _f(sp.area),
        _f(sp.skeletal_length),
        _f(spu.spine_volume_to_spine_area(sp)),
        bbox[0], bbox[1], bbox[2],
    ]

    head_exist = spu.head_exist(sp)

    # --- голова: только если найдена (иначе head_* падают/бессмысленны -> NaN) ---
    if head_exist:
        hbbox = _bbox3(sp.head_bbox_oriented_side_lengths)
        head_vals = [
            _f(sp.head_volume),
            _f(sp.head_area),
            _ratio(sp.head_volume, sp.head_area),
            _f(sp.head_skeletal_length),
            _f(sp.head_n_faces),
            _f(sp.head_width),
            _f(sp.head_width_ray),
            _f(sp.head_width_ray_80_perc),
            _f(sp.head_sdf),
            hbbox[0], hbbox[1], hbbox[2],
        ]
        n_heads = _f(sp.n_heads)
    else:
        head_vals = [nan] * 12
        n_heads = 0.0

    # --- шейка (считается всегда) ---
    nbbox = _bbox3(sp.neck_bbox_oriented_side_lengths)
    neck_vals = [
        _f(sp.neck_volume),
        _f(sp.neck_area),
        _ratio(sp.neck_volume, sp.neck_area),
        _f(sp.neck_skeletal_length),
        _f(sp.neck_n_faces),
        _f(sp.neck_width),
        _f(sp.neck_width_ray),
        _f(sp.neck_width_ray_80_perc),
        _f(sp.neck_sdf),
        nbbox[0], nbbox[1], nbbox[2],
    ]

    values = spine_vals + [1.0 if head_exist else 0.0, n_heads] + head_vals + neck_vals
    assert len(values) == len(FEATURE_NAMES), (len(values), len(FEATURE_NAMES))
    return np.asarray(values, dtype=np.float64), centroid


def _worker(args):
    """Запускается в отдельном процессе. Работает в приватном CWD ради CGAL."""
    path, out_desc, out_centroid, data_type = args
    import numpy as np
    from neurd import parameters

    parameters.params.use(data_type)  # для head/neck это no-op, но ставим честно

    prev_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="spine_cgal_")
    try:
        os.chdir(tmp)  # CGAL пишет временные файлы в "./" -> изолируем
        t0 = time.time()
        vec, centroid = _compute_descriptors(path)
        os.chdir(prev_cwd)  # вернуться ДО записи результата (out-пути абсолютные)
        np.save(out_desc, vec)
        np.save(out_centroid, centroid)
        return (True, path, f"{time.time() - t0:.1f}s")
    except Exception as e:
        import traceback
        return (False, path, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        try:
            os.chdir(prev_cwd)
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)


def _gather_jobs(input_dir: Path, output_dir: Path, pattern: str,
                 recursive: bool, overwrite: bool):
    files = sorted(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
    jobs, skipped = [], 0
    for f in files:
        if not f.is_file():
            continue
        # В рекурсивном режиме зеркалируем подпапки, чтобы не было коллизий имён.
        rel_parent = f.parent.relative_to(input_dir) if recursive else Path(".")
        out_sub = output_dir / rel_parent
        out_desc = out_sub / f"{f.stem}.npy"
        out_centroid = out_sub / f"{f.stem}_centroid.npy"
        if (not overwrite) and out_desc.exists() and out_centroid.exists():
            skipped += 1
            continue
        out_sub.mkdir(parents=True, exist_ok=True)
        jobs.append((str(f), str(out_desc), str(out_centroid)))
    return jobs, skipped, len(files)


def main():
    ap = argparse.ArgumentParser(
        description="Извлечение дескрипторов шипиков (+центроид) из мешей в .npy, параллельно.")
    ap.add_argument("input_dir", type=str, help="Папка с мешами шипиков")
    ap.add_argument("output_dir", type=str, help="Папка для .npy результатов")
    ap.add_argument("--pattern", type=str, default="*.off", help="Глоб-маска (по умолч. *.off)")
    ap.add_argument("--recursive", action="store_true",
                    help="Обходить подпапки (структура зеркалируется в output)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="Число процессов (по умолч. = число ядер)")
    ap.add_argument("--data-type", choices=["microns", "h01"], default="microns",
                    help="Режим параметров NEURD; для head/neck не влияет (по умолч. microns)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Пересчитывать даже если .npy уже есть")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        print(f"[!] не папка: {input_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Зафиксировать порядок фич рядом с результатами.
    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as fh:
        json.dump(FEATURE_NAMES, fh, ensure_ascii=False, indent=2)

    jobs, skipped, n_found = _gather_jobs(
        input_dir, output_dir, args.pattern, args.recursive, args.overwrite)

    print(f"[i] вход: {input_dir} (pattern={args.pattern}, recursive={args.recursive})")
    print(f"[i] найдено мешей: {n_found}, к обработке: {len(jobs)}, пропущено (готовы): {skipped}")
    print(f"[i] workers={args.workers}, data_type={args.data_type}")
    print(f"[i] дескрипторов в векторе: {len(FEATURE_NAMES)} -> {output_dir/'feature_names.json'}")
    if not jobs:
        print("[i] нечего делать.")
        return

    task_args = [(p, od, oc, args.data_type) for (p, od, oc) in jobs]
    failed_log = output_dir / "failed.txt"
    ok = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(_worker, ta) for ta in task_args]
        for i, fut in enumerate(as_completed(futures), 1):
            success, path, msg = fut.result()
            name = Path(path).name
            if success:
                ok += 1
                print(f"[{i}/{len(futures)}] ok  {name}  ({msg})")
            else:
                fail += 1
                short = msg.splitlines()[0] if msg else ""
                print(f"[{i}/{len(futures)}] FAIL {name}  -> {short}")
                with open(failed_log, "a", encoding="utf-8") as fl:
                    fl.write(f"{path}\t{msg}\n\n")

    print(f"\nГотово за {time.time() - t0:.1f}s. ok={ok}, fail={fail}, skipped={skipped}")
    if fail:
        print(f"[i] детали ошибок: {failed_log}")


if __name__ == "__main__":
    main()
