# Рабочее состояние модулей и тестов

Документ для следующих сессий: какие модули в каком состоянии, что
покрыто тестами, какие баги/смеллы известны.

См. также: [DEPS_PLAN.md](DEPS_PLAN.md) — план/состояние зависимостей.

---

## Инфраструктура тестов

**Прогон локальный, без Docker. Python 3.12 + numpy 2.**

```bash
pytest tests/unit/        # → 81 passed, 1 skipped
```

- Установка: [scripts/install_local.sh](scripts/install_local.sh) (Python 3.10–3.12).
- Активация шимов: [tests/unit/__init__.py](tests/unit/__init__.py)
  `skip_if_datasci_tools_unusable()` импортирует `neurd` первым, что включает
  numpy/ipyvolume-шимы из [neurd/__init__.py](neurd/__init__.py).
- `conftest` = только `matplotlib.use("Agg")`.

### Структура `tests/unit/`
```
__init__.py                       guard + neurd shim activation
test_env_compat.py                5 тестов на шимы Фазы 3
test_numpy_compat.py              4 теста на numpy 2 compat
test_mesh_tools_compat.py         4 теста на mesh-стек (open3d/meshparty)
test_core_imports.py              19 тестов: импорт всех 17 модулей core clump
                                  + graph_filters + PRE-1 assert (регресс-гард)
leaves/
  conftest.py                     matplotlib Agg
  test_volume_utils.py            5 тестов
  test_microns_graph_query_utils.py  9 тестов
  test_parameter_utils.py         36 тестов
```

### Маппинг distribution-name → import-name (для requirements/setup)
```
datasci_stdlib_tools     → datasci_tools
machine_learning_tools   → machine_learning_tools
graph_nx_tools           → graph_nx_tools
mesh_processing_tools    → mesh_tools
neuron_morphology_tools  → neuron_morphology_tools
```

---

## Покрытие тестами

| Модуль | Тестов | Уровень |
|---|---|---|
| `volume_utils.py` | 5 | юнит, полный |
| `microns_graph_query_utils.py` | 9 | юнит, полный |
| `parameter_utils.py` | 36 | юнит, расширенный |
| Core clump (`test_core_imports.py`) | 19 | импорт-smoke, регресс-гард PRE-1/2/3 |
| Smoke-тесты Фазы 3 (env/numpy/mesh_tools) | 13 | проверка шимов + версий |
| **Итого** | **82 собрано, 81 passed, 1 skipped** | — |

### parameter_utils.py — что покрыто
- `Parameters` / `PackageParameters` — все основные методы (init, attr, item, update, copy).
- Pure helpers (`_clean_modules_dict`, `_jsonable_dict`, `_add_global_name_to_dict`,
  `parameter_config_folder`, `_this_directory`).
- `attr_map` (вкл. регресс на B3).
- `modes_global_param_and_attributes_dict_from_module` (6 кейсов с фейк-модулем).
- `category_param_from_module` (3 кейса).
- `parameters_from_filepath` (6 кейсов с фейковым `.py`-конфигом в `tmp_path`).
- `set_parameters_for_directory_modules_from_obj` (2 end-to-end с фейк-пакетом
  на диске).

### Пробелы
- Скипается 1 тест в `test_env_compat`: `test_ipyvolume_submodule_resolves_via_stub` —
  реальный `ipyvolume` подтягивается через `meshparty`, stub-путь не активен.
  Stub-механика покрыта параллельно через `test_real_package_wins_over_stub`.
- Core clump покрыт **только импорт-smoke** (`test_core_imports.py`, 19 тестов) —
  гарантирует чистый импорт на голой системе, но **не функционально**. PRE-1
  починен, блокер снят. Функциональных/стадийных тестов на сам pipeline пока нет
  (нужен тестовый меш + тяжёлые зависимости).

---

## Состояние модулей

### Без изменений / в нормальной форме
| Модуль | LOC | Заметка |
|---|---|---|
| `volume_utils.py` | 100 | Баг `nucleus_ids → nuclues_ids` починен. Возможно формализовать как `Protocol` (§7.3 NEURD_STRUCTURE.md), но не приоритет. |
| `microns_graph_query_utils.py` | 113 | Полный rewrite, self-import убран. Покрыт 9 тестами. |

### Сильно изменены / основная работа
| Модуль | LOC | Заметка |
|---|---|---|
| `parameter_utils.py` | 877 | Багфиксы B1/B2/B4/B7, B3 (`.removesuffix`), B5/B6 (`exec`/`eval` → `importlib`). Косметика S1/S3/S4/S5/S8/S11. E1: `json_utils` → stdlib `json` + локальные хелперы. 36 тестов. |

### Удалены целиком
- `functional_tuning_utils.py` (тонкие обёртки над `nu.cdist`)
- `gnn_embedding_utils.py`, `gnn_cell_typing_utils.py`
- `connectome_utils.py`, `connectome_analysis_utils.py`, `connectome_query_utils.py`
- `motif_utils.py`
- `proximity_utils.py`, `proximity_analysis_utils.py`
- `neurd/legacy/` (целый каталог, 4026 LOC)
- `cave_interface.py`, `dandi_utils.py`, `nwb_utils.py`, `ais_utils.py`,
  `motif_null_utils.py`
- `cave_client_utils.py`, `vdi_microns_cave.py`, `nature_paper_plotting.py`
- `parameter_configs/*_old.py`
- `docker/` (вся папка)

### Не трогали — core clump (17 модулей с self-imports)
`neuron_utils`, `proofreading_utils`, `error_detection`, `spine_utils`,
`axon_utils`, `apical_utils`, `preprocess_neuron`, `neuron_visualizations`,
`soma_extraction_utils`, `concept_network_utils`, `branch_utils`, `limb_utils`,
`neuron_searching`, `neuron_statistics`, `classification_utils`,
`cell_type_utils`, `synapse_utils`.

---

## Известные смеллы / отложенные пункты

### parameter_utils.py

| ID | Описание | Почему ждёт |
|---|---|---|
| **B6 loop** | `for i in range(0, 2)` в `set_parameters_for_directory_modules_from_obj`. Второй проход (`plus_unused=True`) затирает результаты первого через `setattr`. Однопроходный эквивалент с `plus_unused=True` идентичен по анализу. | Без интеграционного теста на цепочку `set_volume_params` → реальные модули не верифицируется. Trigger: equivalence-тест с 3+ модулями со взаимными `global_parameters_dict_*`-ссылками. |
| **S6** | `PackageParameters.module_attr_map`: при отсутствующем `module_name` молча возвращает `{}`. Опечатка в имени = тихая работа на дефолтах. | Менять поведение — добавлять warning или strict-mode — нужно решать с пользователем. |
| **S7** | `Parameters.attr_map` возвращает `dict` или `list` в зависимости от флагов. Непоследовательный тип. | Контракт надо обсудить (`dataclass` / `namedtuple` / dict-only). |
| **S9** | `PackageParameters.__setitem__` принимает что угодно, хотя ожидается `Parameters`. | Простая правка `Parameters(v)`, но меняет поведение в крайних случаях. |
| **S10** | Коллизия `Parameters.dict` (property) с возможным ключом `"dict"` в данных. | Косметика — задокументировать. |
| **A1** | `Parameters` совмещает 3 модели доступа (dict/attr/`attr_map`). Кандидат на `pydantic.BaseModel`. | Большая работа. См. [DEPS_PLAN.md Шаг D](DEPS_PLAN.md). |
| **A2** | Конфигурирование через мутацию атрибутов модулей (`set_parameters_for_directory_modules_from_obj`) — главный механизм глобального состояния NEURD. | Развязка требует Phase 5. |

### Pre-existing (не наша работа) — все обойдены
| ID | Где | Что | Статус |
|---|---|---|---|
| **PRE-1** | `neurd/neuron_searching.py:2306` | `fcu` использовался до импорта (line 2370). Блокировал чистый импорт core clump. | ✅ Импорт `fcu` поднят в шапку модуля. |
| **PRE-2** | `datasci_tools/dj_utils.py:13` (upstream) | `raise e(...)` где `e` — caught instance, не класс → `TypeError` без datajoint. | ✅ Обойдён: `dj_utils` лениво импортируется в `spine_utils` только в `dj`-ветках. |
| **PRE-3** | `datasci_tools/dotmotif_utils.py:73` (upstream) | Тот же паттерн `raise e(...)`, вскрылся после PRE-1. | ✅ Обойдён: `dotmotif_utils` лениво в `graph_filters.graph_filter_adapter`. |

---

## Статус шагов (детали — [DEPS_PLAN.md §6](DEPS_PLAN.md))

1. **Шаг A** ✅ — PRE-1 починен + core-import smoke-тесты. Заодно обойдены PRE-2/3.
2. **Шаг B** ✅ — `cave_client_utils`/`vdi_microns_cave`/`nature_paper_plotting` удалены.
3. **Шаг C** ⏸️ — расцепление `cell_type_utils` отложено: `e_i_classification` вплетён
   в pipeline (питает axon-классификацию), удаление меняет поведение. Не блокер импорта.
4. **Шаг D** — частично: чистка ~675 строк мёртвого modsetter-кода сделана;
   премисса «убрать module_utils» оказалась устаревшей (живых вызовов 4, datasci_tools
   остаётся). Полный pydantic-рефактор — низкий приоритет, без сокращения зависимостей.
5. **Шаг E** — E1 ✅ (`json_utils` → stdlib `json`); E2 (`dsu.DictType`) отложен
   (деликатно, нулевая выгода по зависимостям).

### Дальше (Фаза 2): тестируемость pipeline
Картировать стадии конвейера (decimation → soma → skeletonization →
spine/axon/compartments → proofreading), документировать по естественным границам,
добавить характеризационные тесты на существующих швах. Реальное разделение кода —
только если шов чистый. Decoupling core clump — осторожно и точечно.

---

## Конвенции, установленные в этой работе

- **Не трогаем поведение без тестов.** Особенно core clump.
- **Регрессионный тест перед/одновременно с фиксом бага.**
- **Логические правки и косметика — разными коммитами.**
- **Префикс `_` — после `grep`-проверки на внешних callers** (в `*.py` и
  `*.ipynb` отдельно — ноутбуки активно зовут публичный API).
- **Self-import (`from . import X as X`) убирать всегда** — антипаттерн.
- **Не трогать upstream-пакеты** (`datasci_tools`, `mesh_tools`, ...) — наш
  форк, мы не управляем их PyPI.
- **Любую двусмысленность поведения уточнять у пользователя** до изменений.
