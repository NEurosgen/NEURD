# NEURD — оптимизация сегментации: текущее состояние

Цель: сократить время и RAM пути `mesh → Neuron`, **замеряя** (measure-first), а не по интуиции.
Геометрия пайплайна — [PIPELINE.md](PIPELINE.md), карта модулей — [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md).

> Это сжатая сводка durable-выводов. Подробный журнал сессий (профили, тупики, пересмотры) жил
> здесь раньше и убран — историю смотри в `git log` и в `memory/`.

**Целевой кейс:** одно-нейронные меши с **ровно одной сомой** (microns и h01). Оптимизации это
предполагают. Валидационные меши: малый h01 `Applications/.../neuron_2530864375.off` (507k граней,
~125–215s) и большой h01 `1830470325` (3.27M граней). ⚠️ Committed fixture `864691…` — **два**
нейрона, нерепрезентативен.

**Инструмент замера:** [tests/tools/benchmark_neuron.py](tests/tools/benchmark_neuron.py) — точно
повторяет `process_all_neurons`, даёт per-stage wall + RSS-пик + cProfile (пишет отчёт даже при краше).
Гейт fidelity — [tests/integration/test_segmentation_pipeline.py](tests/integration/test_segmentation_pipeline.py).

---

## Текущая конфигурация пайплайна (дефолты)

| Узел | Дефолт | Переключатель |
|---|---|---|
| **Spines** | **ВКЛ** — реальный CGAL SDF-сегментатор (`cgal_Segmentation_Module` .so) | `calculate_spines=False` |
| CGAL teasar-скелет (MAP, толстые ветви) | реальный C++ ext (`calcification_param_Module`) | тихий откат на meshparty если меш non-watertight |
| Poisson (сома) | **no-op** (meshlab-фильтр и так был сломан) | `NEURD_REAL_POISSON=1` → pymeshlab/open3d |
| Decimator (сома) | **open3d** in-process | `NEURD_MESHLAB_DECIMATE=1` → старый meshlab |
| FillHoles (сома) | no-op | — |
| trimesh `_cache` | чистится после сегментации + на ветках | — |
| IDX-TRACE (диагностика декомпозиции) | OFF | `NEURD_IDX_TRACE=1` |

> ⚠️ Ранний вывод «отказ от шипиков» (2026-05-31) **ОТМЕНЁН**: KMeans-заглушка заменена настоящим
> CGAL SDF-сегментатором (2026-06-02, `cgal/cgal_segmentation/`), шипики снова детектируются.
> `_cgal_segmentation.py` (KMeans) остаётся лишь fallback'ом, если .so не собран.

---

## Отгруженные выигрыши

| Что | Эффект | Коммит |
|---|---|---|
| **Poisson → in-process no-op** (meshlab-фильтр был сломан, выход=вход) | малый −4.5× (688→151s) | `469f492` |
| **Decimator → open3d in-process** | большой −184s (−14%) | `fcc72ce` |
| **trimesh cache cleanup** (vertex_adjacency_graph 1.6GB + ветки 1.4GB) | пик RAM −1.1GB (−16%) | `0969fd1` |
| **Объём сомы → convex_hull** (был `fill_mesh_holes_with_fan` ~46s) | малый −35s | `7918ffc` |
| RAM: убран двойной deepcopy + очистка preprocessed_data/кэшей | live-size 458→78 MB | `fedde35`/`2a86407`/`3643d3e` |
| CGAL teasar-скелетонизатор пересобран (был `NameError`) | разблокировал MAP/толстые ветви | `c9d3f7f` |
| CGAL SDF-сегментатор восстановлен (был KMeans-стенд) | вернул шипики | `e53769a` |
| `_rebuild_limb_frames` — фикс рассинхрона кадров стичинга | масса нейронов перестала падать (class A) | `50b769f` |

Poisson/FillHoles были байт-идентичными no-op (доказано захватом пар вход/выход), но платили ~536s
за спавн xvfb+meshlabserver. Замены behaviour-preserving (характеризационный тест).

---

## Где реально время и RAM (замерено)

**Время.** Доминанта — **скелетонизация + граф-обвязка** (teasar-ядро само дёшево 0.31s; дорого
построение networkx-графов скелета) + meshlab-сабпроцессы. На большом нейроне ~60% времени —
**внутри mesh_tools**: `split_by_vertices` ~241s, `resolve_empty`/`filter_face_coloring` ~170s,
`np.unique` 1.2M вызовов ~113s.

**НЕ бочтлнеки (measure-first отсёк бесполезное):** SDF-лучи 0.46s; deepcopy ~2s; смена алгоритма
скелетона (teasar 0.31s — Kimimaro/Skeletor не помогут); GPU для одного нейрона; `max_somas`
short-circuit (на одно-сомном меше пропускать нечего). Параллелизация шипиков через `fork` —
**отложена**: fork-after-threads deadlock (нативные BLAS/CGAL-пулы), а `forkserver`/`spawn` → RAM ×3.

**RAM (большой нейрон).** Driver — НЕ deepcopy (113MB), а **lazy-кэш trimesh**:

| держатель | RAM | чистим? |
|---|---|---|
| `vertex_adjacency_graph` полного меша | 1.6 GB | ✅ `_drop_trimesh_caches` после сегментации |
| кэши 153 branch-мешей | ~1.4 GB | ✅ `_clear_mesh_caches` (на retained) |
| транзиент в `preprocess_limb` (submesh'и плотного лимба) | ~1-2 GB | частично (в mesh_tools) |
| embree BVH + импорты библиотек (open3d/embree/trimesh/nx) | ~1 GB | нет (нативное) |

⚠️ Время прогона коррелирует с **числом лимбов** (run-to-run недетерминизм стичинга = ±100-150s),
не путать со штрафом оптимизаций.

---

## Принцип: НЕ правим mesh_tools напрямую

mesh_tools = plain site-packages (НЕ в репо, НЕ editable): правки теряются при `pip install` (ровно
так уже терялись C++ CGAL `.so` и патчи). 116 функций / 16.7K строк ядра алгоритмов — заменить =
переписать NEURD и потерять fidelity. **Вместо** этого перехватываем дорогие узлы in-process
монкипатчем в `neurd/__init__.py` (durable + обратимо через env-флаг + fidelity-нейтрально): так
сделаны Poisson, FillHoles, Decimator. Трогать функцию mesh_tools можно, только если правка
тривиальна И оформлена как монкипатч в нашем `__init__.py` (а не правка site-packages).

---

## Остаточные durable-рычаги (по убыванию отдачи)

1. **Interior filter** (`tu.remove_mesh_interior`, ~104s) — последний meshlab-сабпроцесс в соме-пути
   (Ambient Occlusion, рендер из 128 ракурсов). In-process замены нет (нужен open3d RaycastingScene).
   Качество сомы пользователю не критично → выполнимо, риск средний. **Рычаг №1 по времени.**
2. **Poisson depth** (только при `NEURD_REAL_POISSON=1`): depth=8 — рабочая точка (−27% от depth=9,
   те же 5 лимбов), depth=7 роняет детекцию сомы.

Всё прочее durable по времени ≈ выжато; пол — реальный compute (скелетонизация + mesh_tools).
**Открытый вопрос:** нужен ли реальный Poisson другим сложным H01, или починка CGAL сняла исходный
краш «not just one mesh» (на `1830470325` он проходит с Poisson=no-op). Прогнать 1-2 H01 прежде чем
считать real Poisson ненужным; guard `correspondence_1_to_1` (raise) всё ещё без try/except.

---

## ⚠️ Известная проблема корректности — для дальнейшей модификации

**Class-A: рассинхрон кадров при стичинге floating-кусков.** `_stitch_floating_pieces` дописывает
floating- и cut-ветки с `branch_face_idx` в кадре *чужого* меша → масса нейронов падала
`IndexError: index N is out of bounds for size N` в `apply_adaptive_mesh_correspondence_to_neuron`.

- ✅ **Починено** (`50b769f`): `_rebuild_limb_frames` после стичинга пересобирает самосогласованный
  кадр (limb_mesh = `combine_meshes(ветки)`, contiguous `branch_face_idx`) для сломанных лимбов.
  Только сломанные (сшитые) лимбы трогаются → нулевой риск для не-сшитых нейронов. Краш устранён,
  нейроны сегментируются. Детали — [PIPELINE.md §2.1](PIPELINE.md) + `memory/class_a_frame_desync.md`.
- 🔧 **Осталось копнуть (глубже, отдельный баг):** сам стичинг плодит **перекрывающиеся** ветки →
  пересобранный лимб-меш раздувается (×1.8–12.7) и adaptive-уточнение на нём пропускается (class-B
  guard, меш дисконнектный). Не краш, но качество: дубль-геометрия + нет 2-hop refinement. Чинить —
  в `attach_floating_pieces_to_limb_correspondence`: перемап индексов при вставке + дедуп overlap.
- **Диагностика:** `NEURD_IDX_TRACE=1` включает реперы N1/N2/N3 (печать + `/tmp/neurd_diag/idx_trace.log`)
  — чистота партиции по этапам. Repro: `tests/integration/reproduce_2889815798.py`.

---

## GPU — честная оценка

Для **одного нейрона** GPU мало даст (узкое место — скелет-графы/топология, не числодробление; SDF
уже 0.46s). Реальный потенциал — **throughput**: `process_all_neurons` гоняет много мешей; после
снижения RAM/нейрон можно батчить больше параллельно. GPU-able по частям: ray-casting/SDF (open3d
RaycastingScene / warp / OptiX), KMeans (cuML). Скелетонизация и stitching — нет. Стек CUDA-Python
не установлен; браться **после** in-process замен, когда ясно — в latency или throughput упёрлись.
