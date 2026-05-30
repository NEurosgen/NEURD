# NEURD — структура и состояние форка (актуально 2026-05-30)

Навигационная карта slim-форка. Источник истины — код; этот файл собирает то, что
нужно знать на старте сессии. Связанные доки: [PIPELINE.md](PIPELINE.md) (как работает
пайплайн + compat-патчи), [OPTIMIZATION.md](OPTIMIZATION.md) (профиль + план оптимизации,
ветка `optimize_segmentation`), [REFACTOR_PLAN.md](REFACTOR_PLAN.md) +
[DEAD_CODE_REACHABILITY.md](DEAD_CODE_REACHABILITY.md) (дальнейшая чистка).

**Цель форка:** оставить только путь сегментации меша `mesh → Neuron` (сомы + лимбы/ветви
с mesh+skeleton + сырые шипики). Всё downstream (autoproof, cell typing, синапсы, аксон,
connectome/motif/proximity, GNN, визуализация, cloud/dataset-адаптеры) удалено.

**Размер:** `neurd/` — **16 файлов, ~24.4k LOC** (со старта ~55k). Граф внутренних
импортов — **DAG, 0 циклов** (см. ниже).

---

## Окружение и тесты

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate neurd   # Python 3.12, numpy 2
python -m pytest tests/unit/                  # fast-gate: 59 passed, 1 skipped (~4 c)
python -m pytest tests/integration/test_segmentation_pipeline.py   # характеризационный: 4 passed (~11 мин на main; ~2.5 мин на optimize_segmentation после Poisson-no-op)
```

- **Fast-gate** (`tests/unit/`): import-smoke всех core-модулей + `parameter_utils` (36) +
  Phase-3 шимы (env/numpy/mesh_tools). `tests/unit/__init__.py` импортирует `neurd` первым →
  активирует numpy/ipyvolume-шимы из [neurd/__init__.py](neurd/__init__.py). 1 skip:
  `test_ipyvolume_submodule_resolves_via_stub` (реальный ipyvolume тянется через meshparty).
- **Характеризационный тест** — единственный backstop для правок ядра: реальная декомпозиция
  fixture-меша (`tests/fixtures/864691135510518224.off`, data_type="microns") через
  `segmentation_pipeline`, пинит контракт `mesh → Neuron(somas + limbs/branches[.mesh+.skeleton])`.
  Шеллит `xvfb-run meshlabserver` (нужны оба бинаря: `apt install meshlab xvfb`); chdir в tmp.
- **CGAL-оракул** `tests/integration/test_cgal_segmentation_oracle.py` — пинит питон-stub CGAL
  против эталона `tests/990_mesh*` (SDF-корреляция ≥0.85; stub даёт ~0.95). Скип без провайдера.

---

## Карта модулей (16 файлов)

| Модуль | LOC | Назначение |
|---|---|---|
| [preprocess_neuron.py](neurd/preprocess_neuron.py) | 4842 | Декомпозиция меша → лимбы/ветви (скелетонизация) |
| [neuron_utils.py](neurd/neuron_utils.py) | 3485 | Утилиты нейрона. Самый импортируемый (9 модулей); теперь intra-package **sink** |
| [neuron.py](neurd/neuron.py) | 3403 | Классы `Neuron`, `Limb`, `Branch`, `Soma` |
| [spine_utils.py](neurd/spine_utils.py) | 3380 | Обнаружение шипиков |
| [neuron_statistics.py](neurd/neuron_statistics.py) | 1977 | Скелетная статистика, расстояния (хаб #2, импортируется 7) |
| [soma_extraction_utils.py](neurd/soma_extraction_utils.py) | 1833 | Идентификация сомы |
| [neuron_searching.py](neurd/neuron_searching.py) | 1777 | Query-система поиска веток (строки→`ns.<fn>` через `@run_options`) |
| [concept_network_utils.py](neurd/concept_network_utils.py) | 1325 | Граф-утилиты концепт-сети |
| [branch_utils.py](neurd/branch_utils.py) | 782 | Утилиты ветвей |
| [parameter_utils.py](neurd/parameter_utils.py) | 697 | Загрузка JSON/py-конфигов параметров |
| [width_utils.py](neurd/width_utils.py) | 457 | Расчёт ширины ветвей из SDF |
| [__init__.py](neurd/__init__.py) | 233 | Compat-патчи + шимы (numpy2/trimesh4/meshlab) — см. PIPELINE.md |
| [limb_utils.py](neurd/limb_utils.py) | 124 | Утилиты лимбов |
| [_cgal_segmentation.py](neurd/_cgal_segmentation.py) | 61 | Python-stub для CGAL-сегментации |
| [segmentation_pipeline.py](neurd/segmentation_pipeline.py) | 59 | Slim-оркестратор (2 стадии) |
| version.py | 0 | — |

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

Со старта ~55k LOC / 44 файла удалено ~30k за фазы: Docker→локальный numpy2/trimesh4 +
8 compat-багфиксов (Фазы 1–4); удаление downstream-кластера целиком — synapse/axon/apical/
визуализация, затем 20 файлов autoproof/typing/graph/dataset-адаптеров + neuron_simplification
(Фазы 5–7, 44→16 файлов); zero-ref dead-code во всех core-файлах (Фаза 9, −~10.7k); устранение
import-циклов + свежий zero-ref (2026-05-30, см. REFACTOR_PLAN.md). Детали удалений — в `git log`.
</content>
