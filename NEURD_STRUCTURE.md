# NEURD — Структура модулей (актуально на 2026-05-29)

Карта текущего состояния форка после Фаз 1–5.
Источник истины — сам код; этот файл — навигационная карта.

См. также: [DEPS_PLAN.md](DEPS_PLAN.md), [PIPELINE.md](PIPELINE.md).

**Текущий размер:** `neurd/` — 16 файлов `*.py`, ~27.4k строк (после Фазы 9 dead-code).
**Unit-тесты:** 59 passed, 0 failed, 1 skipped. **Характеризационный тест:** 4 passed.

---

## 1. Внешние зависимости

### 1.1. Базовые (`requirements.txt`)
```
numpy>=2,<3   scipy        pandas>=2    networkx>=3   matplotlib>=3.7
h5py          tqdm         scikit-learn>=1.3
trimesh>=4    meshparty>=2.0
```

### 1.2. Авторские форк-пакеты
```
datasci-stdlib-tools     → datasci_tools
machine-learning-tools   → machine_learning_tools
graph-nx-tools           → graph_nx_tools
mesh_processing_tools    → mesh_tools
neuron_morphology_tools  → neuron_morphology_tools
code_structure_tools     → code_structure_tools
```

### 1.3. Опциональные
- `[connectome]` — `datajoint`, `python-dotenv` (опциональные `vdi_*`/cloud-адаптеры)
- `datajoint`/`seaborn`/`ipyvolume` не импортируются на top-level ни в одном модуле (soft/optional)

### 1.4. `datasci_tools` — «бог-пакет» автора
Используется почти везде (~22 подмодуля: numpy через `numpy_dep`, pandas, matplotlib, networkx, system_utils и т.д.). Полностью убрать нельзя; шимы для numpy≥2 — в [neurd/__init__.py](neurd/__init__.py).

---

## 2. Ядро (core-модули, чистые зависимости)

Эти модули **не импортируют** кластерные файлы. Именно они работают на slim-пути.

### A. Структура нейрона
| Модуль | LOC | Назначение |
|---|---|---|
| [neuron.py](neurd/neuron.py) | 3455 | Классы `Neuron`, `Limb`, `Branch`, `Soma` |
| [neuron_utils.py](neurd/neuron_utils.py) | 5834 | Утилиты нейрона (самый импортируемый) |
| [branch_utils.py](neurd/branch_utils.py) | 1610 | Утилиты ветвей (Фаза 9: было 2339) |
| [limb_utils.py](neurd/limb_utils.py) | 742 | Утилиты лимбов (Фаза 9: было 1610) |
| [concept_network_utils.py](neurd/concept_network_utils.py) | 1758 | Граф-утилиты концепт-сети (было 2175) |
| [width_utils.py](neurd/width_utils.py) | 548 | Расчёт ширины ветвей |
| [parameter_utils.py](neurd/parameter_utils.py) | 877 | Загрузка JSON/py-конфигов |

### B. Препроцессинг, сома, шипики
| Модуль | LOC | Назначение |
|---|---|---|
| [preprocess_neuron.py](neurd/preprocess_neuron.py) | 5193 | Декомпозиция меша на лимбы/ветви |
| [soma_extraction_utils.py](neurd/soma_extraction_utils.py) | 1786 | Идентификация сомы |
| [spine_utils.py](neurd/spine_utils.py) | 3419 | Обнаружение шипиков |

### C. Поиск и статистика
| Модуль | LOC | Назначение |
|---|---|---|
| [neuron_searching.py](neurd/neuron_searching.py) | 1816 | Поиск ветвей по критериям |
| [neuron_statistics.py](neurd/neuron_statistics.py) | 2180 | Скелетная статистика, расстояния |

### D. Вспомогательные
| Модуль | LOC | Назначение |
|---|---|---|
| [segmentation_pipeline.py](neurd/segmentation_pipeline.py) | — | Slim-оркестратор |
| [_cgal_segmentation.py](neurd/_cgal_segmentation.py) | — | Python-stub для CGAL (Фаза 4) |
| [__init__.py](neurd/__init__.py) | 233 | Compat-патчи + шимы |
| [version.py](neurd/version.py) | — | Версия пакета |

---

## 3. Slim-пайплайн сегментации (как работает сейчас)

Каноничный путь — [segmentation_pipeline.py](neurd/segmentation_pipeline.py),
`mesh → Neuron(somas + limbs/branches[mesh+skeleton])`. **2 стадии:**

1. **Идентификация сомы** — `sm.soma_indentification(mesh)`.
2. **Декомпозиция** — `neuron.Neuron(mesh=...)` + `.calculate_decomposition_products()`
   (внутри — `preprocess_neuron`: скелетонизация, ветви, сырые шипики).

`process_all_neurons.py` (точка входа пользователя) идёт ещё короче —
`neuron.Neuron(mesh=...)` напрямую в воркер-процессе на каждый OFF-меш.

Опущено намеренно (Фаза 7 — урезаны стадии 3–5): уточнение ширины,
упрощение ветвления (`neuron_simplification` удалён), упаковка шипиков
head/neck/shaft, а также multi-soma split, синапсы, E/I cell typing,
axon labeling, auto-proofreading, after-proof статистика.

---

## 4. Удалённые файлы (Фазы 1–6)

### Фаза 6 — удаление кластера (2026-05-29)
Удалены 20 изолированных кластерных модулей + 2 «мёртвых» core-файла.
Core-модули их не импортировали; gate `pytest tests/unit/` остался зелёным.

- **Proofreading / error det.:** `proofreading_utils` (8025), `error_detection` (5277),
  `graph_filters` (1211), `graph_error_detector` (1297), `graph_error_detector_dendrite` (930),
  `graph_error_detector_axon` (~163), `graph_filter_pipeline` (910),
  `neuron_graph_lite_utils` (956), `neuron_pipeline_utils` (625)
- **Cell typing:** `classification_utils` (2784), `cell_type_utils` (1889)
- **Multi-soma / прочее:** `soma_splitting_utils` (295), `neuron_geometry_utils` (218)
- **Dataset-адаптеры:** `vdi_default` (1208), `vdi_microns`, `vdi_h01`, `volume_utils` (101),
  `microns_volume_utils` (698), `microns_graph_query_utils` (113), `h01_volume_utils` (586)
- **Мёртвый код (0 импортёров):** `branch_attr_utils` (182, функции инлайнены в `spine_utils`),
  `documentation_utils` (43)
- **Тесты:** удалены `tests/unit/leaves/test_{volume_utils,microns_graph_query_utils}.py`
  (тестировали удалённые модули) и `tests/integration/test_{autoproof,segmentation}_pipeline.py`
  (импортировали `vdi_microns`/`neuron_pipeline_utils`).

### Фаза 5 (2026-05-29)
- `synapse_utils.py` (~4.5k LOC) — синапс-функции
- `axon_utils.py` (~3.9k LOC) — аксон-классификация
- `apical_utils.py` (~1.8k LOC) — apical/basal-классификация
- `neuron_visualizations.py` (~4k LOC) — визуализация

### Фазы 1–4 (~16k+ LOC)
- **Connectome:** `connectome_utils`, `connectome_analysis_utils`, `connectome_query_utils`
- **Motif:** `motif_utils`, `motif_null_utils`
- **Proximity:** `proximity_utils`, `proximity_analysis_utils`
- **GNN:** `gnn_embedding_utils`, `gnn_cell_typing_utils`
- **Cloud/IO:** `cave_interface`, `cave_client_utils`, `vdi_microns_cave`, `dandi_utils`, `nwb_utils`, `ais_utils`, `functional_tuning_utils`, `nature_paper_plotting`
- **Legacy:** `neurd/legacy/` (4026 LOC), `parameter_configs/*_old.py`
- **Docker:** вся папка `docker/`, `docs/` (устаревший Sphinx)

---

## 5. Карта зависимостей core-модулей

**Самые импортируемые:**
| Модуль | Импортируется из (примерно) |
|---|---|
| `neuron_utils` | branch_utils, concept_network_utils, limb_utils, spine_utils, neuron_searching, neuron_statistics, neuron.py, … |
| `neuron_statistics` | concept_network_utils, neuron_searching, spine_utils, neuron.py |
| `neuron_searching` | neuron_utils, branch_utils, concept_network_utils, spine_utils |
| `branch_utils` | neuron_utils, spine_utils |

**Точки входа (не импортируются другими core-модулями):**
- `segmentation_pipeline.py` — slim-оркестратор
- `__init__.py` — пакет-инициализатор

---

## 6. Следующие шаги

1. ✅ **Характеризационный тест** есть — [tests/integration/test_segmentation_pipeline.py](tests/integration/test_segmentation_pipeline.py)
   (fixture-меш → somas/limbs/branches+skeleton). Зафиксировал 4 регрессии Фаз 5–7 (исправлены).
2. **Замена `meshlabserver`** на in-process (open3d/trimesh) — Decimator, Poisson, FillHoles, Interior.
   Теперь безопасно: контракт декомпозиции запинен тестом.
3. **Оптимизация памяти** — `Branch.__init__` делает deepcopy submesh (главный драйвер RAM).
4. ✅ **Dead-code срезан во всех core-файлах** (Фаза 9, оба захода, −~10.7k LOC до ~27.4k).
   Zero-ref дальше нет. Осталось: распутывание клубка (18 циклов module-load) — это рефактор,
   не удаление; и более глубокий reachability-анализ (функции, живые только из мёртвых веток).
