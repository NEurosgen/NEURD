# NEURD — структура и состояние форка (актуально 2026-07-22)

Навигационная карта slim-форка. Источник истины — код; этот файл собирает то, что
нужно знать на старте сессии. Связанные доки: [PIPELINE.md](PIPELINE.md) (как работает
пайплайн + compat-патчи), [OPTIMIZATION.md](OPTIMIZATION.md) (профиль + план оптимизации),
[REFACTOR_PLAN.md](REFACTOR_PLAN.md) + [DEAD_CODE_REACHABILITY.md](DEAD_CODE_REACHABILITY.md)
(дальнейшая чистка), [MESH_OPERATIONS_LAYER.md](MESH_OPERATIONS_LAYER.md) +
[MESH_OPS_PHASE_B_PLAN.md](MESH_OPS_PHASE_B_PLAN.md) (owned mesh-ops слой / seam).

**Цель форка:** оставить только путь сегментации меша `mesh → Neuron` (сомы + лимбы/ветви
с mesh+skeleton + сырые шипики). Всё downstream (autoproof, cell typing, синапсы, аксон,
connectome/motif/proximity, GNN, визуализация, cloud/dataset-адаптеры) удалено.

**Размер:** `neurd/` — **19 файлов, ~16.8k LOC** (со старта ~55k). Граф внутренних
импортов — **DAG, 0 циклов** (см. ниже).

---

## Окружение и тесты

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate neurd   # Python 3.12, numpy 2
python -m pytest tests/unit/                  # fast-gate (~4 c); incl. tests/unit/test_submesh_ops.py
python -m pytest tests/integration/test_segmentation_pipeline.py -s  # характеризационный (~2 мин на h01-нейроне; -s печатает путь сохранённой сегментации)
```

- **Fast-gate** (`tests/unit/`): import-smoke всех core-модулей + `parameter_utils` +
  `submesh_ops` `tu`-эквивалентность + Phase-3 шимы (env/numpy/mesh_tools). `tests/unit/__init__.py` импортирует `neurd` первым →
  активирует numpy/ipyvolume-шимы из [neurd/__init__.py](neurd/__init__.py). 1 skip:
  `test_ipyvolume_submodule_resolves_via_stub` (реальный ipyvolume тянется через meshparty).
- **Характеризационный тест** — единственный backstop для правок ядра: реальная декомпозиция
  h01-нейрона (`Applications/Tutorials/Auto_Proof_Pipeline/neuron_2530864375.off`, одно-сомный,
  fixture ставит `params.use("h01")`) через `segmentation_pipeline`, пинит контракт
  `mesh → Neuron(somas + limbs/branches[.mesh+.skeleton])`. После проверок сохраняет результат
  (`save_segmentation` из `process_all_neurons.py`) в `tests/integration/_seg_output/<stem>/`
  для визуального осмотра — best-effort, путь настраивается `SEG_OUTPUT_DIR=` (gitignored, ~30МБ
  .off). Шеллит `xvfb-run meshlabserver` (нужны оба бинаря: `apt install meshlab xvfb`); chdir в tmp.
- **CGAL-оракул** `tests/integration/test_cgal_segmentation_oracle.py` — пинит питон-stub CGAL
  против эталона `tests/990_mesh*` (SDF-корреляция ≥0.85; stub даёт ~0.95). Скип без провайдера.

---

## Карта модулей (19 файлов)

| Модуль | LOC | Назначение |
|---|---|---|
| [spine_utils.py](neurd/spine_utils.py) | 3262 | Обнаружение шипиков (COLD — не в `segmentation_pipeline`; ~51 `tu.*`, крупнейший holdout) |
| [preprocess_neuron.py](neurd/preprocess_neuron.py) | 3150 | Декомпозиция меша → лимбы/ветви; `preprocess_neuron`/`preprocess_limb` декомпозированы на именованные фазы (см. PIPELINE.md §2) |
| [neuron_utils.py](neurd/neuron_utils.py) | 2599 | Утилиты нейрона. Самый импортируемый; intra-package **sink** |
| [neuron.py](neurd/neuron.py) | 1817 | Классы `Neuron`, `Limb`, `Branch`, `Soma` |
| [neuron_searching.py](neurd/neuron_searching.py) | 1292 | Query-система поиска веток (строки→`ns.<fn>` через `@run_options`) |
| [soma_extraction_utils.py](neurd/soma_extraction_utils.py) | 1045 | Идентификация сомы |
| [concept_network_utils.py](neurd/concept_network_utils.py) | 849 | Граф-утилиты концепт-сети |
| [parameter_utils.py](neurd/parameter_utils.py) | 697 | Загрузка JSON/py-конфигов параметров |
| [__init__.py](neurd/__init__.py) | 356 | Compat-патчи + шимы (numpy2/trimesh4/meshlab) — см. PIPELINE.md |
| [branch_utils.py](neurd/branch_utils.py) | 355 | Утилиты ветвей |
| [neuron_statistics.py](neurd/neuron_statistics.py) | 347 | Скелетная статистика, расстояния |
| [submesh_ops.py](neurd/submesh_ops.py) | 312 | **Owned** partition/provenance слой (`SubMesh`) — см. MESH_OPS доки |
| [width_utils.py](neurd/width_utils.py) | 183 | Расчёт ширины ветвей из SDF |
| [parameters.py](neurd/parameters.py) | 155 | Явные microns/h01 параметры (заменили global-config monkey-patching) |
| [_mesh_ops.py](neurd/_mesh_ops.py) | 138 | L6 mesh-примитивы (decimate/poisson/fill_holes, in-process open3d) |
| [_correspondence_backend.py](neurd/_correspondence_backend.py) | 80 | **Seam** для хрупких `cu`/`m_sk` вызовов + единственный недетерминизм (rng seam) |
| [_cgal_segmentation.py](neurd/_cgal_segmentation.py) | 66 | Python-stub для CGAL-сегментации |
| [segmentation_pipeline.py](neurd/segmentation_pipeline.py) | 63 | Slim-оркестратор (2 стадии) |
| version.py | 0 | — |

**Seam-семейство** (`submesh_ops.py` / `_mesh_ops.py` / `_correspondence_backend.py`): NEURD-owned
слой, изолирующий/заменяющий хрупкую поверхность `mesh_tools`. Детали — в MESH_OPS доках.

**Точки входа (никто из core не импортирует):** `segmentation_pipeline.py`,
`process_all_neurons.py` (реальный entry пользователя — `neuron.Neuron(mesh=)` в воркере).

### Граф импортов = DAG (после рефактора 2026-05-30)
Раньше: god-hub `neuron_utils` + 6 двусторонних циклов, band-aid'ами через bottom-of-file
импорты. Сейчас **0 циклов любой длины**: `neuron_utils` сделан intra-package sink'ом
(3 ссылки на `neuron`/`cnu`/`nst` → локальные импорты в местах вызова), 7 self-import'ов
убраны, мёртвые/одноразовые перекрёстные импорты сняты/лазифицированы. Перепроверка —
скрипт обхода `^from \. import` / `^from .X import` (в REFACTOR_PLAN.md §1).

---

## Конвенции (соблюдать)

- **Не менять поведение без теста.** Backstop для ядра — характеризационный тест (11 мин).
  Fast-gate ловит только syntax/import.
- **Self-import (`from . import X as X`) — антипаттерн, всегда убирать** (заменить на прямые
  вызовы; если нужен сам модуль-объект — `sys.modules[__name__]`).
- **Логические правки и косметика — разными коммитами.** Коммит-сообщения с why.
- **Перед удалением функции — grep на живых вызывающих** в `neurd/`+`tests/`+`process_all_neurons.py`.
  ⚠️ Ноутбуки в `Applications/` игнорировать — это тюториалы, пользователь их не использует.
- **Не трогать upstream-пакеты** (`datasci_tools`, `mesh_tools`, `meshparty`) — нет контроля
  над их PyPI; правки только обёрткой/патчем в `neurd/`.

### Известные смеллы `parameter_utils.py` (ждут интеграционного теста на `set_volume_params`)
- **B6 loop**: `for i in range(0,2)` в `set_parameters_for_directory_modules_from_obj` —
  второй проход затирает первый (избыточно, но не блокирует).
- **S6**: `PackageParameters.module_attr_map` при отсутствующем `module_name` молча → `{}`.
- **S7**: `Parameters.attr_map` возвращает `dict` или `list` по флагам (контракт обсудить).

---

## Зависимости (кратко)

- **База** (`requirements.txt`): `numpy>=2,<3`, scipy, `pandas>=2`, `networkx>=3`,
  `matplotlib>=3.7`, h5py, tqdm, `scikit-learn>=1.3`, `trimesh>=4`, `meshparty>=2`.
- **Авторские форки** (PyPI): `datasci-stdlib-tools`→`datasci_tools` (бог-пакет, ~22 подмодуля,
  numpy через `numpy_dep` — убрать нельзя), `mesh_processing_tools`→`mesh_tools`, и др.
- **Рантайм-бинари:** `meshlabserver` + `xvfb-run` (mesh-операции шеллятся). Цель — заменить
  на in-process (см. PIPELINE.md).
- Опционально/soft: `datajoint`, `seaborn`, `ipyvolume` — не импортируются на top-level.

---

## История (сжато)

Со старта ~55k LOC → ~16.8k. Фазы: Docker→локальный numpy2/trimesh4 + 8 compat-багфиксов;
удаление downstream-кластера целиком (synapse/axon/apical/визуализация, autoproof/typing/graph/
dataset-адаптеры + neuron_simplification, 44→16 файлов); zero-ref dead-code (−~10.7k) + устранение
import-циклов; затем читаемость-рефактор (декомпозиция крупных функций/классов, −verbose) и
mesh-ops/correspondence seam (submesh_ops/_correspondence_backend). Детали — в `git log` и `memory/`.
</content>
