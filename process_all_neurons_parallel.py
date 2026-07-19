# process_all_neurons_parallel.py
# -*- coding: utf-8 -*-
#
# Параллельная версия process_all_neurons.py.
#
# Логика обработки одного меша (NEURD-сегментация, спайны через CGAL,
# раскладка на диск, манифест) ПОЛНОСТЬЮ переиспользуется из
# process_all_neurons.py — здесь только планировщик, который запускает
# несколько воркеров одновременно.
#
# Главное ограничение (ради чего всё затевалось):
#   суммарный размер входных мешей, обрабатываемых ОДНОВРЕМЕННО, держится
#   ниже бюджета памяти (по умолчанию 1 ГБ). Размер входного .off берётся
#   как прокси к пику RAM — на больших мешах алгоритм раздувается и без
#   этого ограничения несколько крупных мешей разом съедают всю оперативку.
#
# Правила планировщика:
#   * новый воркер стартует, только если (sum размеров уже запущенных мешей
#     + размер кандидата) <= бюджет;
#   * меш, который сам по себе больше бюджета, не отбрасывается — он ждёт,
#     пока освободятся все слоты, и запускается в одиночку;
#   * параллелизм дополнительно ограничен числом воркеров (--max-workers,
#     по умолчанию = число ядер);
#   * очередь отсортирована по размеру (по возрастанию): из оставшихся сразу
#     набирается пул мелких мешей под бюджет (greedy first-fit) — они идут
#     параллельно, а крупные оседают в хвост и обрабатываются по одному.
#     Без сортировки крупный меш вперемешку с мелкими занимал бы почти весь
#     бюджет и не давал мелким идти параллельно.

import os
import gc
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any

# Те же ограничения численных бэкендов, что и в последовательной версии —
# выставляем ДО импорта numpy/neurd (process_all_neurons тоже их ставит).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Переиспользуем всю «рабочую» часть последовательного скрипта.
from process_all_neurons import (
    _worker_process,
    ensure_neurd_defaults,
    read_manifest,
    append_manifest_atomic,
    _rss_mb,
    EXPORT_EXT,
    DATA_TYPE,
)

# Бюджет памяти по умолчанию: суммарный размер одновременных мешей < 1 ГБ.
DEFAULT_MEM_BUDGET_MB = 1024


def _mb(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


def process_folder_parallel(
    input_dir: str,
    output_dir: str,
    manifest: str = "processed.txt",
    pattern: str = "*.off",
    no_save: bool = False,
    no_decimate: bool = False,
    max_workers: int = None,
    mem_budget_mb: int = DEFAULT_MEM_BUDGET_MB,
    poll_interval: float = 0.2,
) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest) if os.path.isabs(manifest) else (out_dir / manifest)

    ensure_neurd_defaults()

    files = sorted(in_dir.glob(pattern))
    if not files:
        print(f"[!] no files matched: {in_dir}/{pattern}")
        return

    done = read_manifest(manifest_path)
    done_failed = read_manifest(out_dir / "failed.txt")

    # Список «к обработке» с размерами файлов, уже без сделанного/упавшего.
    pending: List[Dict[str, Any]] = []
    skipped = 0
    for f in files:
        sid = f.stem
        if sid in done or sid in done_failed:
            skipped += 1
            continue
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        pending.append({"path": f, "spine_id": sid, "size": size})

    # Сортируем очередь по размеру (по возрастанию). Тогда try_launch() сразу
    # набирает пул из мелких мешей под бюджет и гонит их параллельно, а крупные
    # оседают в хвост и идут по одному. Иначе крупный меш, стоящий вперемешку
    # с мелкими, выбирался бы первым, занимал почти весь бюджет — и мелкие
    # рядом не помещались, обрабатываясь фактически в один поток.
    pending.sort(key=lambda j: j["size"])

    budget_bytes = int(mem_budget_mb) * 1024 * 1024
    if not max_workers or max_workers <= 0:
        max_workers = os.cpu_count() or 1

    rss_limit_mb = int(os.environ.get("SEG_RSS_LIMIT_MB", "0"))
    per_item_timeout = int(os.environ.get("SEG_TIMEOUT_SEC", "0"))
    failed_log = out_dir / "failed.txt"

    total = len(pending)
    print(f"[i] data_type={DATA_TYPE}")
    print(f"[i] manifest: {manifest_path} (loaded {len(done)} ids, skipping {skipped})")
    print(f"[i] found {len(files)} mesh(es), {total} to process in: {in_dir}")
    print(f"[i] parallel: max_workers={max_workers}, mem_budget={mem_budget_mb} MB "
          f"({_mb(budget_bytes):.0f} MB)")
    if rss_limit_mb:
        print(f"[i] per-worker RSS limit: {rss_limit_mb} MB")
    if per_item_timeout:
        print(f"[i] per-item timeout: {per_item_timeout} s")

    oversized = [j for j in pending if j["size"] > budget_bytes]
    if oversized:
        print(f"[!] {len(oversized)} mesh(es) exceed the budget by themselves — "
              f"each will run ALONE (no co-tenants):")
        for j in oversized:
            print(f"      {j['path'].name}: {_mb(j['size']):.0f} MB")

    running: List[Dict[str, Any]] = []
    processed = failed = started = 0

    def in_flight_bytes() -> int:
        return sum(j["size"] for j in running)

    def try_launch() -> None:
        """Запускаем столько воркеров, сколько влезает в бюджет и слоты."""
        nonlocal started
        while pending and len(running) < max_workers:
            cur = in_flight_bytes()
            pick = None
            for idx, j in enumerate(pending):
                if not running:
                    # Ничего не крутится — стартуем самый мелкий из оставшихся
                    # (очередь отсортирована по возрастанию); если он больше
                    # бюджета, поедет в одиночку.
                    pick = idx
                    break
                if cur + j["size"] <= budget_bytes:
                    # Очередь отсортирована: первый влезающий = самый мелкий
                    # подходящий, что максимизирует число задач в пуле.
                    pick = idx
                    break
            if pick is None:
                # Ни один из оставшихся не влезает в текущий остаток бюджета —
                # ждём, пока освободятся слоты.
                break

            job = pending.pop(pick)
            parent_conn, child_conn = mp.Pipe(duplex=False)
            p = mp.Process(
                target=_worker_process,
                args=(str(job["path"]), str(out_dir), (not no_decimate),
                      EXPORT_EXT, child_conn),
                daemon=False,
            )
            p.start()
            # Закрываем записывающий конец в родителе: иначе fd течёт на
            # каждой задаче и не детектится EOF на pipe.
            child_conn.close()

            started += 1
            job.update(proc=p, conn=parent_conn, start=time.time())
            running.append(job)
            print(f"[start {started}/{total}] {job['path'].name} "
                  f"({_mb(job['size']):.0f} MB) | in-flight: "
                  f"{_mb(in_flight_bytes()):.0f}/{mem_budget_mb} MB, "
                  f"workers: {len(running)}/{max_workers}")

    def finalize(job: Dict[str, Any]) -> None:
        nonlocal processed, failed
        p = job["proc"]
        conn = job["conn"]
        if "result" in job:
            ok, msg = job["result"]
        else:
            # Процесс завершился сам — забираем сообщение из pipe.
            try:
                if conn.poll(1.0):
                    ok, msg = conn.recv()
                else:
                    ok, msg = (p.exitcode == 0), f"exitcode={p.exitcode}"
            except EOFError:
                ok, msg = (p.exitcode == 0), f"exitcode={p.exitcode}"
        try:
            conn.close()
        except Exception:
            pass
        p.join()

        if ok:
            if not no_save:
                append_manifest_atomic(manifest_path, job["spine_id"])
            processed += 1
            print(f"    [ok] {job['spine_id']}")
        else:
            failed += 1
            with open(failed_log, "a", encoding="utf-8") as fl:
                fl.write(f"{job['spine_id']}\t{msg}\n")
            print(f"    [x] failed: {job['spine_id']}: {msg}")
        gc.collect()

    try:
        while pending or running:
            try_launch()

            finished: List[Dict[str, Any]] = []
            for job in running:
                p = job["proc"]
                conn = job["conn"]

                # Дренируем pipe, пока процесс ещё жив: большой traceback может
                # не влезть в буфер и заблокировать воркер на send().
                try:
                    if "result" not in job and conn.poll(0):
                        job["result"] = conn.recv()
                except (EOFError, OSError):
                    pass

                if not p.is_alive():
                    finished.append(job)
                    continue

                now = time.time()
                if per_item_timeout and (now - job["start"] > per_item_timeout):
                    p.terminate(); p.join(5)
                    job["result"] = (False, f"Timeout after {per_item_timeout}s")
                    finished.append(job)
                    continue
                if rss_limit_mb:
                    r = _rss_mb(p.pid)
                    if r is not None and r > rss_limit_mb:
                        p.terminate(); p.join(5)
                        job["result"] = (False,
                                         f"RSS limit exceeded: {r} MB > {rss_limit_mb} MB")
                        finished.append(job)
                        continue

            for job in finished:
                running.remove(job)
                finalize(job)

            # Спим, только если ничего не завершилось (иначе сразу досыпаем слоты).
            if running and not finished:
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[!] interrupted — terminating running workers...")
        for job in running:
            try:
                job["proc"].terminate()
                job["proc"].join(5)
            except Exception:
                pass
        raise

    print(f"\nDone. processed={processed}, skipped={skipped}, failed={failed}")
    print(f"[i] manifest now has {len(read_manifest(manifest_path))} ids at {manifest_path}")


# ---------------------- CLI ----------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Параллельная NEURD-сегментация мешов (H01) с бюджетом памяти.")
    ap.add_argument("input_dir", type=str, help="Папка с мешами (.off/.ply/...)")
    ap.add_argument("output_dir", type=str,
                    help="Папка для сегментаций (каждая как исходный basename)")
    ap.add_argument("--pattern", type=str, default="*.off",
                    help="Глоб-маска входных файлов")
    ap.add_argument("--manifest", type=str, default="processed.txt",
                    help="Путь к манифесту (по умолчанию: <output_dir>/processed.txt)")
    ap.add_argument("--no-save", action="store_true",
                    help="Не отмечать в манифесте (dry-run манифеста)")
    ap.add_argument("--no-decimate", action="store_true", help="Отключить decimation")
    ap.add_argument("--max-workers", type=int, default=int(os.environ.get("SEG_MAX_WORKERS", "0")),
                    help="Максимум одновременных воркеров (0 = число ядер)")
    ap.add_argument("--mem-budget-mb", type=int,
                    default=int(os.environ.get("SEG_MEM_BUDGET_MB", str(DEFAULT_MEM_BUDGET_MB))),
                    help="Бюджет суммарного размера одновременных мешей, МБ (по умолч. 1024)")
    args = ap.parse_args()

    process_folder_parallel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest=args.manifest,
        pattern=args.pattern,
        no_save=args.no_save,
        no_decimate=args.no_decimate,
        max_workers=args.max_workers,
        mem_budget_mb=args.mem_budget_mb,
    )

# Пример запуска (бюджет 1 ГБ, до 4 воркеров):
# python process_all_neurons_parallel.py /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/notebooks/H01 /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/notebooks/H01_Seg --max-workers 8 --mem-budget-mb 1024
# Для микронс python process_all_neurons_parallel.py /mnt/wwn-0x50014ee26c2ca7b0-part1/minnie65_meshes /mnt/wwn-0x50014ee26c2ca7b0-part1/minnie65_seg_part_2 --max-workers 8 --mem-budget-mb 1024