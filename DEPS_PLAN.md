# NEURD — Рефакторинг зависимостей (Фазы 1–5)

Цель: **оставить только то, что нужно для сегментации меша** — на выходе
`Neuron` с сомами + ветками (mesh + skeleton) + шипиками. Всё autoproof / cell typing /
синапсы / статистика / визуализация удалено.

Слим-оркестратор: [neurd/segmentation_pipeline.py](neurd/segmentation_pipeline.py).
Геометрия pipeline: [PIPELINE_GEOMETRY.md](PIPELINE_GEOMETRY.md).

---

## ✅ Выполнено (все фазы)

### Фазы 1–4 (до 2026-05-28)
- Docker → локальный Python 3.12 / numpy 2 / trimesh 4.
- Удалены GNN, connectome, motif, proximity (~16k LOC), `extras_require`, numpy-шим.
- 8 compat-багфиксов (CGAL→питон-stub, meshlab OFF, multi-soma guard, pykdtree-1D, in1d, dotmotif).
- Пайплайн зелёный 9/9 стадий; добавлен slim-оркестратор.

### Фаза 5 — удаление downstream-кластера (2026-05-29)

Выполнена полностью. Стратегия: сначала прорежение ядра, потом удаление кластера.

| Волна | Что сделано |
|---|---|
| **plot/nviz** | Удалены все `nviz`-импорты и ~60 `if plot_xxx` блоков из 11 core-файлов |
| **5.2** synapse_utils | 17 wrapper-функций из neuron_searching + syu-код из 4 файлов; `synapse_utils.py` удалён |
| **5.3** axon/apical/class | ~20 функций из 6 файлов, дефолты `au.axon_width` → `None`; `axon_utils.py`, `apical_utils.py` удалены |
| **5.1+5.4** error_det + nst | 18 wrapper-функций из neuron_searching, мёртвые функции из 3 файлов; `error_detection` импорты убраны из ядра |
| **5.5** branch_attr_utils | 5 функций инлайнены в `spine_utils.py`; `branch_attr_utils.py` оставлен (чистый, без внешних dep) |
| **nviz файл** | `neuron_visualizations.py` удалён |

**Результат:** `tests/unit/` → **74 passed, 0 failed, 1 skipped**.
Все 11 core-файлов импортируются чисто без кластерных зависимостей.

---

## Фаза 6 — удаление кластера (2026-05-29)

Выполнена. Все 20 изолированных кластерных файлов удалены, плюс 2 «мёртвых»
core-файла (0 импортёров) и 4 тест-файла, завязанных на удалённое.
Gate `pytest tests/unit/` остался зелёным (**60 passed, 1 skipped**).

**Метод проверки перед удалением:** временно убрали все 20 файлов из `neurd/`,
прогнали импорт точки входа (`process_all_neurons` / `segmentation_pipeline`) и
`pytest tests/unit/`, затем вернули и удалили штатно через `git rm`.

| Удалено | LOC | Группа |
|---|---|---|
| `proofreading_utils.py` | 8025 | autoproof |
| `error_detection.py` | 5277 | autoproof |
| `classification_utils.py` | 2784 | typing |
| `cell_type_utils.py` | 1889 | typing |
| `graph_error_detector.py` | 1297 | graph-proofreading |
| `graph_filters.py` | 1211 | graph-proofreading |
| `vdi_default.py` | 1208 | dataset-адаптер |
| `graph_error_detector_dendrite.py` | 930 | graph-proofreading |
| `graph_filter_pipeline.py` | 910 | graph-proofreading |
| `neuron_graph_lite_utils.py` | 956 | graph |
| `neuron_pipeline_utils.py` | 625 | старый оркестратор |
| `h01_volume_utils.py` | 586 | dataset-адаптер |
| `soma_splitting_utils.py` | 295 | multi-soma |
| `neuron_geometry_utils.py` | 218 | геометрия |
| `microns_graph_query_utils.py` | 113 | dataset-адаптер |
| `microns_volume_utils.py` | 698 | dataset-адаптер |
| `graph_error_detector_axon.py` | ~163 | graph-proofreading |
| `vdi_microns.py`, `vdi_h01.py` | ~170 | dataset-адаптеры |
| `volume_utils.py` | 101 | абстрактный DataInterface |
| `branch_attr_utils.py` | 182 | мёртвый (инлайнен в spine_utils) |
| `documentation_utils.py` | 43 | мёртвый (0 импортёров) |

**Нюанс:** `volume_utils` и `microns_graph_query_utils` имели юнит-тесты в
`tests/unit/leaves/` — их пришлось удалить вместе с модулями, иначе gate падал на
коллекции (`ImportError`). Также удалены интеграционные `test_autoproof_pipeline.py`
и `test_segmentation_pipeline.py` (импортировали `vdi_microns`/`neuron_pipeline_utils`).

После Фазы 6 в `neurd/` осталось **17 файлов `*.py`** — все на slim-пути сегментации.

---

## Фаза 7 — урезка стадий 3–5 пайплайна (2026-05-29)

Пользователю нужны только soma + декомпозиция (`Neuron(mesh=)`), как и делает
`process_all_neurons.py`. Стадии 3–5 оркестратора (width refinement / branch
simplification / spine head-neck-shaft packaging) не нужны.

- [segmentation_pipeline.py](neurd/segmentation_pipeline.py) урезан до 2 стадий
  (соме + декомпозиция); убраны импорты `bu`, `nsimp`, `spu` и параметр
  `branch_simplification`.
- **Удалён `neuron_simplification.py`** (684 LOC) — его импортировал только
  `segmentation_pipeline` (стадия 4). `refine_width_…` и `add_head_neck_shaft_spine_objs`
  остаются в `branch_utils`/`spine_utils` (это core, ф-ции используются внутри них).

`neurd/*.py`: 17 → **16**. Gate: **59 passed, 1 skipped**.

---

## Фаза 8 — характеризационный тест + регрессии декомпозиции (2026-05-29)

Добавлен [tests/integration/test_segmentation_pipeline.py](tests/integration/test_segmentation_pipeline.py):
реальная декомпозиция fixture-меша → контракт `mesh → Neuron(somas + limbs/branches[.mesh+.skeleton])`.
**4 passed (~11 мин, реальный `xvfb-run meshlabserver`).**

Тест вскрыл 4 регрессии от удалений Фаз 5–7 (все исправлены в этом же коммите):

| # | Симптом | Причина | Фикс |
|---|---|---|---|
| 1 | `AttributeError: glia_nuclei_faces_from_mesh` | ф-ция удалена в `ee336ed`, но вызов жив в `neuron.py:2246` | восстановлена в `soma_extraction_utils.py` (чистая, только `tu.*`) |
| 2 | `ValueError: expected 6, got 5` (создание сомы) | `soma_synapses` убрали из `zip`, но `curr_synapses` остался в распаковке | убран dangling `curr_synapses` (`neuron.py:2516`) |
| 3 | `axon_length not found` / `KeyError: n_boutons` | `nru.axon_*` удалены (Фаза 5), но используются в `neuron_stats` | выкинуты `axon_length/axon_area/n_boutons` из stats + `stats_to_ignore` |
| 4 | `NameError: nst is not defined` | self-вызовы `nst.<local>` без in-function self-import | заменены на прямые вызовы (по конвенции «без self-import») |

Также: `segmentation_pipeline` больше не зовёт `calculate_decomposition_products`
(stats-шаг, не нужен для контракта; зеркалит `process_all_neurons`). Скелеты лежат
на объекте уже после декомпозиции.

**Вывод:** slim-пайплайн `mesh → Neuron` теперь воспроизводимо зелёный end-to-end.

---

## Фаза 9 — dead-code внутри 4 больших файлов (2026-05-29)

Статическое удаление мёртвых функций под защитой характеризационного теста.
Метод: функция удаляется, если её имя **не встречается нигде** в `neurd/` + `tests/` +
`process_all_neurons.py` — ни как ссылка, ни как строка (значит, её не дёрнуть и через
`eval`/`getattr` query-систему, которая иначе засветила бы имя). Защищены имена под
динамическую сборку: `*_limb_ns`, `all_*_branches_from_branches`, `compute_*`,
`current_*`, `endpoint_*`, `arg*`, `synapses_*`. Итерация до фикспоинта (каждый проход
вскрывает функции, чьи единственные вызыватели удалены прошлым проходом).

| Файл | LOC до | LOC после |
|---|---|---|
| `neuron_utils.py` | 9584 | 5834 |
| `spine_utils.py` | 6442 | 3419 |
| `neuron_statistics.py` | 3727 | 2180 |
| `neuron_searching.py` | 2080 | 1816 |
| **Σ (4 файла)** | **21833** | **13249** (−8584, ~39%) |

В основном — остатки удалённых подсистем (synapse/axon/typing/proofreading-хелперы,
`*_old`-варианты, `plot_*` от удалённой визуализации). Gate: unit 59 passed;
характеризационный тест 4 passed (контракт декомпозиции не сломан).

Что НЕ трогали: динамически-диспетчеризуемые query-функции и остальные core-файлы
(`neuron`, `preprocess_neuron`, `branch_utils`, `concept_network_utils`, …) — следующий
заход, если нужно.

---

## Чего НЕ трогаем

- Upstream-пакеты (`datasci_tools`, `mesh_tools`, `meshparty`) — только обёртки/шимы в `neurd/`.
- `meshparty`/CGAL-скелетонизация, `open3d` — ядро mesh-геометрии.
- Ядро декомпозиции: `neuron.py` / `preprocess_neuron` / `soma_extraction_utils` /
  `neuron_simplification` / `branch_utils` / `spine_utils`.

---

## Применённые compat-патчи (numpy 2 / trimesh 4 / мёртвый meshlabserver)

| # | Симптом | Причина | Фикс (где) |
|---|---|---|---|
| 1 | `TypeError ... scalar index` в soma split | trimesh≥4 `mesh.split()` → `list`, не `ndarray` | `soma_extraction_utils.py` |
| 2 | Poisson не выполняется | `meshlab.Poisson` пишет `<xmlfilter>` XML, игнорируется MeshLabServer 2020.09 | `__init__.py`: патч `Poisson.initialize_script_filters` |
| 3 | `NameError: csm` (CGAL не установлен) | C++ расширение CGAL отсутствует | `__init__.py`: stub `_cgal_segmentation.py` (ray_trace SDF + KMeans) |
| 4 | `scipy ValueError: axis 0 index ... exceeds` | MeshLabServer OFF-экспортёр: компактные вершины, грани в старой нумерации | `__init__.py`: патч `Meshlab.fetch_mesh_from_off` |
| 5 | `IndexError ... size 1` в multi-soma split | две сомы на одном стартовом узле лимба | `proofreading_utils.py:1147`: guard |
| 6 | `ValueError: data_pts ... 2 dimensions` | новая pykdtree требует 2D | `__init__.py`: обёртка `skeleton_utils.KDTree` |
| 7 | `numpy_dep has no attribute 'in1d'` | `np.in1d` удалён в numpy 2 | `__init__.py`: `numpy.in1d = isin` |
| 8 | стадия 9: dotmotif | другой грамматич. диалект в mainline | форк reimerlab + stub `Neo4jExecutor` |

---

## Метрики

| Метрика | До (старт) | Фаза 5 | Фаза 6 | Фаза 7 | Фаза 9 |
|---|---|---|---|---|---|
| Файлов `neurd/*.py` | 44 | 39 | 17 | 16 | **16** |
| LOC `neurd/` | ~55k | — | ~38k | ~38k | **~29.6k** |
| Unit-тестов passed | 81 | 74 | 60 | 59 | **59** (1 skipped) |
| Характеризационный тест | — | — | — | — | **4 passed** |
