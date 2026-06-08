# NEURD — slim-пайплайн сегментации: как работает + патчи

Что геометрически происходит на пути `mesh → Neuron`, где это в коде, что можно/нельзя
трогать, и какие compat-патчи держат всё на numpy2/trimesh4 без Docker.
Карта модулей — в [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md).

> ⚠️ Downstream-стадии (multi-soma split, cell typing, axon, autoproof, after-proof stats)
> и их модули **удалены** из форка. Здесь описан только живой slim-путь (2 стадии).

---

## 0. Картина целиком

Вход — **меш одного нейрона** из EM: огромная треугольная поверхность (сотни тысяч граней) —
округлое **тело (сома)** с ветвящимися **отростками** (дендриты/аксон), покрытыми **шипиками**.
Задача slim-пайплайна — превратить «мешок треугольников» в структуру: найти сому, разложить
отростки на **скелет** (центрлайны) и **ветки**, построить **граф связности (concept network)**,
снять сырые шипики. Три примитива, вокруг которых всё крутится: **SDF**, **скелет**, **concept network**.

Каноничный путь — [segmentation_pipeline.py](neurd/segmentation_pipeline.py):
1. **Идентификация сомы** — `sm.soma_indentification(mesh)`.
2. **Декомпозиция** — `neuron.Neuron(mesh=...)` (внутри `preprocess_neuron`: скелетонизация,
   ветви, сырые шипики, concept network).

`process_all_neurons.py` (entry пользователя) идёт ещё короче — `neuron.Neuron(mesh=...)`
напрямую в воркер-процессе на каждый OFF-меш. Stats-шаг `calculate_decomposition_products`
**вне** slim-пути (его единственного вызывателя убрали; частично починен).

---

## 1. Геометрические примитивы

- **SDF (Shape Diameter Function, «толщина»).** Для каждой грани луч **внутрь** меша по
  инвертированной нормали; длина до выхода = локальная толщина. Сома толстая → высокий SDF;
  дендриты/аксон тонкие → низкий. Главный признак «сома vs отростки» и оценки ширины веток.
  **Код:** `mesh_tools.trimesh_utils.ray_trace_distance(mesh)` (embree если есть). ⚠️ В пайплайне
  SDF ожидается **нормализованным в [0,1]** — пороги (`soma_width_threshold=0.32`) под эту шкалу.
- **Скелет (skeleton).** 1D-граф по «середине» трубчатого отростка (осевая линия). Длина =
  длина отростка; точки несут ширину (из SDF); разбивается на **ветки** в точках ветвления.
  **Код:** скелетонизация в `preprocess_neuron.py` (meshafterparty/MAP, `mesh_tools.skeleton_utils`).
- **Mesh segmentation по SDF.** Кластеризация граней по SDF — CGAL MRF graph-cut. Реальный C++
  `cgal_Segmentation_Module` (.so) **восстановлен** (`cgal/cgal_segmentation/`, см. cgal/README.md);
  KMeans-стенд `neurd/_cgal_segmentation.py` остаётся лишь **fallback'ом**, если .so не собран
  (`__init__.py` регистрирует stub только при отсутствии реального модуля). KMeans давал грубую
  сегментацию (ок для сомы, но **0 шипиков**) — отсюда возврат к настоящему CGAL. **Код:**
  `tu.mesh_segmentation(mesh, clusters, smoothness)` → sub-меши + **SDF-медиана сегмента**.
- **Concept network.** Направленный граф веток лимба: узлы=ветки, рёбра=«A продолжается в B»,
  корень = точка касания сомы, направление upstream→downstream (от сомы наружу). Один граф на
  (лимб, сома). **Код:** `nru.branches_to_concept_network(...)`, живёт в `Limb.concept_network`.

### Иерархия объекта Neuron
```
Neuron
 ├── somas:  S0, S1, ...                    (тела: меш + центр)
 └── limbs:  L0, L1, ...                    (отростки = связные компоненты после вырезания сомы)
       ├── branches: 0,1,2,...              (участки скелета между ветвлениями)
       │     ├── mesh, mesh_face_idx        (подмеш ветки)
       │     ├── skeleton                    (центрлайн)
       │     ├── width_array (из SDF)
       │     ├── endpoint_upstream/downstream
       │     └── web                         (меш-«перепонка» в точке ветвления)
       └── concept_network                   (граф веток, корень = касание сомы)
```
Файлы: `neuron.py` (классы), `neuron_utils.py` (`nru.*` — запросы по графу), `branch_utils.py`,
`limb_utils.py`, `concept_network_utils.py`.

---

## 2. Две живые стадии — геометрия и код

### Стадия 1 — Soma identification (`soma_extraction_utils`)
Самая геометрически плотная. Найти меш(и) сомы:
1. Decimation (грубее) → быстрый кандидат (`meshlab.Decimator`).
2. Выделить крупные куски меша.
3. **Poisson surface reconstruction** (`meshlab.Poisson`) → водонепроницаемая оболочка.
4. **Remove interior** (`meshlab.Interior`, через Ambient Occlusion).
5. **Mesh segmentation по SDF** (`tu.mesh_segmentation`, clusters≈3) → сегменты + SDF-медианы.
6. **Отбор:** сегмент проходит если `SDF_median > soma_width_threshold` (0.32) **и** размер в
   `[soma_size_threshold, _max]` → «толстый и крупный округлый кусок».
7. **Sphere validator** (bbox ≈ шар) + **backtrack** грубой оболочки на грани оригинального меша.

⚠️ Порог `0.32` и отбор завязаны на нормализованный [0,1] SDF — любой новый поставщик SDF
**обязан** нормализовать (stub делает, перцентили [2,98]). Узкое место замены — шаги 3–4 (MeshLab).

### Стадия 2 — Decomposition (`neuron.Neuron(...)` → `preprocess_neuron.preprocess_neuron`)

`preprocess_neuron` переписан под **строго одну сому без глии/ядер** и декомпозирован в
тонкий оркестратор поверх именованных фаз (каждая — отдельная функция в
[neurd/preprocess_neuron.py](neurd/preprocess_neuron.py)):

1. `_extract_single_soma(mesh, segment_id)` — ищет ровно одну сому (Фаза 1/2; raise если нет).
2. `_segment_limbs_from_soma(mesh, soma, params)` — вырезает сому → остаток распадается на
   **лимбы** (связные компоненты, касающиеся сомы) + floating-куски. Возвращает
   `soma_to_piece_connectivity` с **позиционными** индексами лимбов `0..N-1` (это важно: limb-
   узлы concept network именуются `L{j}` по этому порядку — рассинхрон → `KeyError: 'data'`).
3. `_decompose_limbs(branch_meshes, touching, params)` — на каждый лимб зовёт `preprocess_limb`:
   скелетонизация (meshafterparty/MAP) → ветки → `limb_correspondence` (branch_mesh,
   branch_skeleton, width_from_skeleton).
4. `_stitch_floating_pieces(...)` — пришивает значимые floating-куски к скелету.
   ⚠️ **Содержит баг рассинхрона кадров — см. §2.1.**
5. `_rebuild_limb_frames(...)` — **фикс §2.1:** пересобирает самосогласованный кадр у лимбов,
   чью партицию стичинг сломал (limb_mesh = `combine_meshes(ветки)`, contiguous `branch_face_idx`).
6. `_build_concept_networks(...)` — concept network на каждый (лимб, сома).

**`preprocess_limb` — контракт параметров.** Сигнатура
`preprocess_limb(mesh, neuron_params, limb_params, soma_touching_vertices_dict=None, ...)`:
- `neuron_params` — общие для нейрона (width/size_threshold_MAP, axon_width_*, adaptive-invalidation,
  mp_only_*). Их читают helper'ы `_decide_next_limb_cfg`/`_cycle_for_something`.
- `limb_params` — параметры одного прохода скелетонизации (invalidation_d, smooth_neighborhood,
  combine/filter meshparty, use_meshafterparty).
- Прочие скаляры — из `parameters.params` (config-driven) либо литералы (бывшие хардкод-дефолты).

Тело `preprocess_limb` декомпозировано по фазам: `_cycle_for_something` (MP-скелетонизация +
adaptive invalidation_d), `_decompose_map_piece` (MAP-кусок, CGAL), `_fix_mp_soma_extension`
(достройка soma-extending веток), Part 17–18 (`_merge_map_mp_correspondence`,
`_rearrange/_clean_network_starting_info`). ⚠️ **MAP/stitching-путь снова АКТИВЕН** после
пересборки CGAL teasar-скелетонизатора (`c9d3f7f`): толстые ветви (`width > width_threshold_MAP`)
идут через `_decompose_map_piece`, floating-куски — через `_stitch_floating_pieces`. Раньше путь был
мёртв (NameError calcification_param) и покрывался только статически — теперь исполняется, и именно
он вскрыл баг §2.1. Конфиг датасета функция **не перебивает** — вызыватель обязан выставить
`parameters.params.use("microns"|"h01")` заранее.

⚠️ Сердце core clump. `Branch.__init__` делает deepcopy submesh/skeleton — дорого по RAM
(перф ниже), но менять рискованно. Скелетонизация (meshparty) — отдельная зависимость, НЕ meshlab.

### Инварианты — НЕ менять без сквозного теста
- Пороги отбора сомы (`soma_width_threshold=0.32`, size thresholds) — завязаны на [0,1] SDF.
- `Branch`/`Limb`/`Neuron` и concept network — на структуре графа/атрибутах держится всё.
- Семантика направления concept network (upstream/downstream от сомы).

### 2.1 ⚠️ Баг стичинга: рассинхрон кадров `branch_face_idx` (class A) — частично починен

**Симптом (массовый).** Множество H01-нейронов падали `IndexError: index N is out of bounds for
axis 0 with size N` в `nru.apply_adaptive_mesh_correspondence_to_neuron`
([neuron_utils.py](neurd/neuron_utils.py), `ex_limb.mesh.submesh([surround_mesh_faces])`).

**Корень (доказан реперами IDX-TRACE, `NEURD_IDX_TRACE=1`).** Инвариант: `branch_face_idx` каждой
ветки обязан адресовать **хранимый лимб-меш** (`limb_meshes[limb_idx]` = `branch_meshes[limb_idx]`)
как чистая партиция. **Декомпозиция (`_decompose_limbs`) этот инвариант держит** (репер N1 чист на
всех лимбах). **Ломает его `_stitch_floating_pieces`** (репер N2 — битые ровно сшитые лимбы):
дописывает floating-ветки (`flaot_data` из `preprocess_limb(mesh=floating_piece)`) и cut-ветки
(`correspondence_1_to_1(mesh=stitch_mesh)`) с `branch_face_idx` **в кадре их собственного меша**, не
перемапленным в кадр лимб-меша → индексы выходят за пределы (`oob`) и/или алиасят чужие грани
(`overlap`). Конкретно — [preprocess_neuron.py](neurd/preprocess_neuron.py), цикл вставки floating-веток
(`limb_correspondence_cp[...][curr_limb_key_len + 1 + float_idx] = flaot_data`). **Не watertight,
не off-by-one, не MAP/MP-комбинация** — все эти гипотезы проверены и отвергнуты.

**Текущий фикс (`50b769f`, шаг 5 выше).** `_rebuild_limb_frames` после стичинга: для лимба, чья
партиция перестала быть чистой, пересобирает кадр — `limb_mesh = tu.combine_meshes(ветки)`,
`branch_face_idx = ` непрерывные диапазоны (`combine_meshes` сохраняет порядок/число граней даже на
дублях — проверено). Только сломанные лимбы трогаются → не-сшитые нейроны не затронуты. **Краш
устранён**, нейроны сегментируются (валидация: `neuron_2889815798`, 7 лимбов, без краша).

**🔧 Что осталось для дальнейшей модификации (глубже).** Фикс делает выход *самосогласованным*, но
не лечит причину: стичинг плодит **перекрывающиеся** ветки → пересобранный лимб-меш раздувается
(×1.8–12.7 на тестовом нейроне), а adaptive-уточнение на нём **пропускается** (рабочий guard
в `apply_adaptive_*`, т.к. склейка веток даёт дисконнектный меш). Не краш, но качество: дубль-геометрия
+ нет 2-hop refinement. Правильное место чинить — `attach_floating_pieces_to_limb_correspondence`:
перемапливать `branch_face_idx` в кадр лимб-меша **при вставке** и дедупить overlap, чтобы
`_rebuild_limb_frames` стал не нужен. Диагностика — `NEURD_IDX_TRACE=1` (реперы N1/N2/N3 →
`/tmp/neurd_diag/idx_trace.log`), repro — `tests/integration/reproduce_2889815798.py`,
детали — `memory/class_a_frame_desync.md`.

**Родственный режим — class B `missing labels was not resolved` (тоже стичинг, теперь защищён).**
Когда стык floating-куска попадает в СЕРЕДИНУ ветки основного лимба, ветка режется и
`correspondence_1_to_1(mesh=stitch_mesh)` заново партиционирует её на 2 куска. При вырожденном
разрезе (одной половине не достаётся связный патч граней) `resolve_empty_conflicting_face_labels`
кидает `missing labels was not resolved` — раньше это валило весь нейрон. **Фикс:** вызов обёрнут в
try/except (cut-путь в `attach_floating_pieces_to_limb_correspondence`); при сбое кусок помечается
обработанным и **пропускается** (floating best-effort), нейрон достраивается. На месте крушения
`limb_correspondence_cp` ещё не мутирован, поэтому пропуск безопасен. ⚠️ **Глубже:** причина —
вырожденный/мид-веточный разрез; чинить там же, где class A (correspondence при вставке).

**Родственный режим — class C `too many indices for array: array is 1-dimensional` (тоже стичинг,
теперь защищён).** При декомпозиции floating-куска (`preprocess_limb(mesh=k)` в шаге 1 attach) его
MAP-скелетонизация (`_decompose_map_piece` → CGAL/meshparty → `mesh_subtraction_by_skeleton`) может
оставить **вырожденный/пустой** leftover-submesh (`faces` формы `(0,)`/1-D), и trimesh падает на
`faces[:, …]` (deep in mesh_tools, править durable нельзя). **Фикс:** вызов `preprocess_limb` в цикле
декомпозиции floating-кусков обёрнут в try/except; кусок, который не декомпозируется, **пропускается**
(не добавляется в `floating_limbs_correspondence` — всё ниже выводится из него, индексы консистентны).
⚠️ **Глубже:** тот же leftover может всплыть и на MAIN-лимбе (в `_decompose_limbs`), где «пропустить»
нельзя — там пока не защищено.

**Режим «разорванный лимб» — `concept graph nodes != branches` (НЕ дубликаты).** На нейроне с
**очень многими** floating-кусками стичинг наваливает на лимб десятки/сотни фрагментов и **не смыкает
их скелеты** (limb 0: декомпозиция дала 23 связные ветки → после стичинга 217, из них ~148 разорваны,
гэпы тысячи единиц, 88 полностью изолированы). `branches_to_concept_network` от сомы достигает лишь
связной компоненты → `len(nodes) != len(branches)` → краш в самом конце (~2ч). Диагностировано
measure-first: `_dump_concept_network_mismatch` (форензик-дамп скелетов + вердикт genuine-dup vs
disconnected) + IDX-TRACE N1↔N2 (показал 23→217). **Правильный фикс (объёмный, НЕ сделан):** перед
concept network оставлять в лимбе только сому-связную компоненту, разорванные фрагменты отбрасывать
(заодно снимет inflation `_rebuild_limb_frames`). **Сделан fail-fast:** сразу после сегментации
считаем значимые floating-куски; при `> NEURD_MAX_FLOATING_PIECES` (дефолт 100, 0=off) — raise в
минутах, а не через 2ч. Эвристика; счётчик печатается всегда (для калибровки).

---

## 3. Применённые compat-патчи (numpy2 / trimesh4 / мёртвый meshlabserver)

Все — в `neurd/` (не upstream). Большинство — monkeypatch'и в
[neurd/__init__.py](neurd/__init__.py), применяются при `import neurd` (до импорта mesh_tools).
**Это load-bearing — понимать назначение перед правкой `__init__.py`.**

| # | Симптом | Причина | Фикс (где) |
|---|---|---|---|
| 1 | `TypeError ... scalar index` (soma split) | trimesh≥4 `mesh.split()` → `list`, не `ndarray` | `soma_extraction_utils.py`: `list(...)` + list-comprehension |
| 2 | Poisson не выполняется, нет выходного файла | `meshlab.Poisson` пишет `<xmlfilter>` XML, MeshLabServer 2020.09 игнорирует | `__init__.py`: **superseded** — `Poisson.__call__` → in-process no-op (выход=вход), т.к. фильтр всё равно no-op на этой сборке; реальный Poisson за `NEURD_REAL_POISSON=1` (open3d/pymeshlab) |
| 3 | `NameError: csm` (CGAL не установлен) | C++ `cgal_Segmentation_Module` отсутствует | **Реальный ext восстановлен** (`cgal/cgal_segmentation/`). `__init__.py` ставит stub `_cgal_segmentation.py` (KMeans) в `sys.modules` **только если .so не найден** (fallback) |
| 4 | `scipy ValueError: axis 0 index ... exceeds` | MeshLabServer 2020.09 OFF-экспортёр: компактные вершины, грани в старой нумерации | `__init__.py`: патч `Meshlab.fetch_mesh_from_off` (перенумерация searchsorted) |
| 5 | `IndexError ... size 1` (multi-soma split) | две сомы на одном стартовом узле → путь из 1 узла | `proofreading_utils.py:1147` guard *(модуль удалён; патч исторический)* |
| 6 | `ValueError: data_pts ... 2 dimensions` | новая pykdtree требует 2D, upstream строит из 1D | `__init__.py`: обёртка `skeleton_utils.KDTree` (1D→(N,1)) |
| 7 | `numpy_dep has no attribute 'in1d'` | `np.in1d` удалён в numpy 2 | `__init__.py`: `numpy.in1d = isin` + `numpy_dep.in1d = isin` |
| 8 | `NameError: calcification_param` (MAP-путь, толстые ветви) | CGAL teasar-скелетонизатор (`calcification_param_Module`) собирался в Docker; с его удалением исчез. `mesh_tools.skeleton_utils` импортит через `try/except` → имя не связано | **Не stub, а реальная пересборка:** [cgal/cgal_skeleton_param/](cgal/cgal_skeleton_param/) — исходник из git, портирован под CGAL 6 (C++17, `IO/OFF.h`, `CGAL::IO::read_OFF`). Собирается `install_local.sh` (best-effort, нужны CGAL/eigen/gmp/mpfr). Без него падают только нейроны с толстыми ветвями |

**Перф-монкипатчи (не баг-фиксы, а замена дорогих узлов mesh_tools in-process — см. [OPTIMIZATION.md](OPTIMIZATION.md)):**

| узел | было | стало (`__init__.py`) | эффект |
|---|---|---|---|
| `meshlab.Poisson` | xvfb+meshlabserver subprocess (но фильтр no-op) | in-process no-op; реальный за `NEURD_REAL_POISSON=1` | −536s (4.5× на малом) |
| `meshlab.FillHoles` | subprocess (тоже сломан → no-op) | in-process pass-through | убран лишний форк |
| `meshlab.Decimator` | subprocess + OFF round-trip | open3d in-process (`_mesh_ops.decimate`); `NEURD_MESHLAB_DECIMATE=1` → старый | **−184s (−14%) на большом** |
| trimesh `mesh._cache` | копится (vertex_adjacency_graph 1.6GB, ветки 1.4GB) | `_drop_trimesh_caches` после сегментации + `_clear_mesh_caches` чистит ветки | **пик RAM −1.1GB (−16%)** |

> Корни багов — в `git log` соответствующих правок. Здесь — что/где, чтобы понимать `__init__.py`.
> Патч #8 (CGAL skeletonizer) — отдельный C++ extension, не monkeypatch; см. [cgal/README.md](cgal/README.md).
> **Почему перф-фиксы идут монкипатчами, а не правкой mesh_tools:** mesh_tools = plain site-packages
> (правки теряются при reinstall, как `.so`/skeleton_utils); 116 функций / 16.7K строк ядра — заменить
> = переписать NEURD. Перехват узлов in-process — durable + обратимо + fidelity-нейтрально. Детально — OPTIMIZATION.md.

---

## 4. Forward: оптимизация — см. [OPTIMIZATION.md](OPTIMIZATION.md)

Полный workstream оптимизации (профиль, отгруженные выигрыши, план, GPU) вынесен в
[OPTIMIZATION.md](OPTIMIZATION.md). Кратко — что **замерено** (профиль 2026-05-30, не гипотезы):

- **MeshLab сабпроцесс ~74% времени.** Из них **Poisson был сломанным no-op** (выход = вход) и
  стоил ~536s — уже заменён in-process pass-through: пайплайн **688s → 151s (4.5×)**.
- **SDF-лучи 0.46s, deepcopy ~2s** — НЕ драйверы времени (прежние «гипотезы» про них неверны;
  deepcopy может быть драйвером RAM, но не скорости).
- Meshlab in-process замены: **Poisson** ✅ (no-op), **FillHoles** ✅ (no-op), **Decimator** ✅
  (→ open3d, −184s). Осталось: **Interior removal** (`tu.remove_mesh_interior` ~104s, реальный
  Ambient-Occlusion фильтр, in-process замены нет — последний крупный рычаг по времени).
- Реальный compute-пол: **скелетонизация** (meshparty/CGAL teasar) + ~60% времени ВНУТРИ
  mesh_tools (`split_by_vertices`, `resolve_empty`, `np.unique`) — durable-правки там невозможны.
- **RAM (2026-06-02): driver = НЕ deepcopy (113MB), а lazy-кэш trimesh** — `vertex_adjacency_graph`
  1.6GB на полном меше + 1.4GB на ветках. Частично закрыто очисткой кэша (пик 6974→5876 MB).
  Карта RAM целиком — OPTIMIZATION.md.
