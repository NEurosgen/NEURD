# NEURD — Структура и карта зависимостей (актуально на 2026-05-28)

Карта текущего состояния форка после стрипа до mesh-segmentation pipeline.
Источник истины по фактам — сам код; этот файл — навигационная карта.

См. также: [DEPS_PLAN.md](DEPS_PLAN.md) — план/лог работы с зависимостями,
[LEAVES_WORK.md](LEAVES_WORK.md) — состояние модулей и тестов.

**Текущий размер:** `neurd/` — 41 файл `*.py`, ~84k строк.

---

## 1. Внешние зависимости

### 1.1. Базовые (`requirements.txt`)
```
numpy>=2,<3   scipy        pandas>=2    networkx>=3   matplotlib>=3.7
h5py          tqdm         scikit-learn>=1.3
trimesh>=4    meshparty>=2.0

# Авторские форк-пакеты (dist-name → import-name)
datasci-stdlib-tools     → datasci_tools
machine-learning-tools   → machine_learning_tools
graph-nx-tools           → graph_nx_tools
mesh_processing_tools    → mesh_tools
neuron_morphology_tools  → neuron_morphology_tools
code_structure_tools     → code_structure_tools
```

### 1.2. Опциональные (`setup.py: extras_require`)
- `[connectome]` — `datajoint`, `python-dotenv` (опциональные `vdi_*`/cloud-адаптеры)
- `[viz]` — `seaborn`, `ipyvolume`
- `[all]` — всё вышеперечисленное

> Группы `[ml]` нет: GNN-блок удалён. `datajoint`/`seaborn`/`ipyvolume` не
> импортируются на top-level ни в одном модуле (soft/optional).

### 1.3. `datasci_tools` — «бог-пакет» автора
Через него идёт почти всё (numpy через `numpy_dep`, pandas, matplotlib, networkx,
system_utils и т.д.) — используется через ~22 подмодуля. Полностью убрать его
нельзя; трогаем только в рамках конкретного модуля. Шимы для numpy≥2 и
optional-cloud-пакетов — в [neurd/__init__.py](neurd/__init__.py).

---

## 2. Логические блоки (текущие модули)

### A. Ядро структуры нейрона
- [neuron.py](neurd/neuron.py) — 4315
- [neuron_utils.py](neurd/neuron_utils.py) — 10292 *(самый импортируемый: in=24)*
- [branch_utils.py](neurd/branch_utils.py) — 1627
- [branch_attr_utils.py](neurd/branch_attr_utils.py) — 182
- [limb_utils.py](neurd/limb_utils.py) — 752
- [concept_network_utils.py](neurd/concept_network_utils.py) — 1840
- [neuron_graph_lite_utils.py](neurd/neuron_graph_lite_utils.py) — 956
- [neuron_geometry_utils.py](neurd/neuron_geometry_utils.py) — 218
- [neuron_simplification.py](neurd/neuron_simplification.py) — 694
- [width_utils.py](neurd/width_utils.py) — 548
- [parameter_utils.py](neurd/parameter_utils.py) — 877 *(загрузка JSON/py-конфигов)*
- [documentation_utils.py](neurd/documentation_utils.py) — 43

### B. Препроцессинг, сома, спайны, синапсы
- [preprocess_neuron.py](neurd/preprocess_neuron.py) — 5364
- [soma_extraction_utils.py](neurd/soma_extraction_utils.py) — 2147
- [soma_splitting_utils.py](neurd/soma_splitting_utils.py) — 295
- [spine_utils.py](neurd/spine_utils.py) — 7022
- [synapse_utils.py](neurd/synapse_utils.py) — 4472

### C. Классификация / поиск / морфология
- [classification_utils.py](neurd/classification_utils.py) — 2785
- [cell_type_utils.py](neurd/cell_type_utils.py) — 1890 *(E/I + cell typing; вплетён в pipeline)*
- [apical_utils.py](neurd/apical_utils.py) — 1805
- [axon_utils.py](neurd/axon_utils.py) — 3925
- [neuron_searching.py](neurd/neuron_searching.py) — 2377
- [neuron_statistics.py](neurd/neuron_statistics.py) — 4374

### D. Proofreading (детекция/исправление ошибок реконструкции)
- [proofreading_utils.py](neurd/proofreading_utils.py) — 8017
- [error_detection.py](neurd/error_detection.py) — 5278
- [graph_filters.py](neurd/graph_filters.py) — 1220

### E. Визуализация
- [neuron_visualizations.py](neurd/neuron_visualizations.py) — 4092

### F. Volume / dataset-адаптеры (microns vs h01)
- [volume_utils.py](neurd/volume_utils.py) — 101 *(абстрактный `DataInterface`)*
- [microns_volume_utils.py](neurd/microns_volume_utils.py) — 698
- [microns_graph_query_utils.py](neurd/microns_graph_query_utils.py) — 113
- [h01_volume_utils.py](neurd/h01_volume_utils.py) — 586
- [vdi_default.py](neurd/vdi_default.py) — 1208
- [vdi_microns.py](neurd/vdi_microns.py) — 64
- [vdi_h01.py](neurd/vdi_h01.py) — 105

### G. Пайплайн / вход
- [neuron_pipeline_utils.py](neurd/neuron_pipeline_utils.py) — 626 *(оркестрация autoproof; точка входа, in=0)*

### H. Параметры
- [parameter_configs/parameters_config_default.py](neurd/parameter_configs/parameters_config_default.py)
- [parameter_configs/parameters_config_h01.py](neurd/parameter_configs/parameters_config_h01.py)
- [parameter_configs/parameters_config_microns.py](neurd/parameter_configs/parameters_config_microns.py)

### I. Орфаны — экспериментальный graph-proofreading (НЕ подключён к API)
Полностью изолированы (in=0, out=0), импортируются разве что лениво:
- [graph_error_detector.py](neurd/graph_error_detector.py) — 1365
- [graph_error_detector_dendrite.py](neurd/graph_error_detector_dendrite.py) — 932
- [graph_error_detector_axon.py](neurd/graph_error_detector_axon.py) — 163
- [graph_filter_pipeline.py](neurd/graph_filter_pipeline.py) — 911 *(in=1, ленивый импорт в proofreading_utils)*

> ~3.4k LOC. Кандидат на удаление после подтверждения, что не нужны pipeline.

---

## 3. Карта зависимостей (внутренние импорты)

**Самые импортируемые (ядро, in-degree):**
| Модуль | in | out |
|---|---|---|
| `neuron_utils` | 24 | 15 |
| `neuron_visualizations` | 22 | 7 |
| `neuron_statistics` | 19 | 11 |
| `neuron_searching`, `synapse_utils` | 16 | 12 / 13 |
| `axon_utils` | 13 | 16 |
| `branch_utils`, `spine_utils` | 11 | 9 / 10 |
| `concept_network_utils`, `proofreading_utils` | 10 | 5 / 20 |

**Точки входа / листья (in-degree 0):** `neuron_pipeline_utils` (out=13, оркестратор),
`neuron_geometry_utils`, `vdi_h01`, `vdi_microns`, `microns_graph_query_utils`,
+ орфаны graph-proofreading (§2.I).

---

## 4. Core clump (17 модулей — не трогать без интеграционных тестов)
`neuron_utils`, `neuron`, `branch_utils`, `concept_network_utils`, `limb_utils`,
`preprocess_neuron`, `proofreading_utils`, `error_detection`, `neuron_visualizations`,
`neuron_statistics`, `axon_utils`, `apical_utils`, `spine_utils`, `synapse_utils`,
`classification_utils`, `cell_type_utils`, `neuron_searching`.

Взаимные импорты + общее глобальное состояние через
`datasci_tools.module_utils.all_modules_set_global_parameters_and_attributes`
(вызывается из `neurd.set_volume_params`). Циклы разорвать дёшево нельзя — нужны
сначала тесты. PRE-1/2/3 починены → ядро теперь чисто импортируется на голой
системе (см. [DEPS_PLAN.md](DEPS_PLAN.md)), есть core-import smoke-тесты.

---

## 5. Что удалено при стрипе (для контекста)
Полностью удалены post-сегментационные подсистемы и dead-код (~16k LOC + ~675
строк закомментированного modsetter-кода):
- **Connectome:** `connectome_utils`, `connectome_analysis_utils`, `connectome_query_utils`
- **Motif:** `motif_utils`, `motif_null_utils`
- **Proximity:** `proximity_utils`, `proximity_analysis_utils`
- **GNN:** `gnn_embedding_utils`, `gnn_cell_typing_utils` (+ extras `[ml]`)
- **Cloud/IO/прочее:** `cave_interface`, `cave_client_utils`, `vdi_microns_cave`,
  `dandi_utils`, `nwb_utils`, `ais_utils`, `functional_tuning_utils`,
  `nature_paper_plotting`
- **Legacy:** `neurd/legacy/` (4026 LOC), `parameter_configs/*_old.py`
- **Docker:** вся папка `docker/`; **`docs/`** (устаревший Sphinx от авторов)

---

## 6. Следующие шаги
План разбиения pipeline на тестируемые стадии и точечные задачи — см.
[DEPS_PLAN.md §6](DEPS_PLAN.md) и [LEAVES_WORK.md](LEAVES_WORK.md).
