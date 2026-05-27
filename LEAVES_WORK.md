# Работа по блоку «листья» (§7.1 из [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md))

Документ для меня (Claude) — рабочее состояние, чтобы при следующем заходе
не перевыполнять разведку.

См. также: [DEPS_PLAN.md](DEPS_PLAN.md) — план поэтапной чистки/замены
зависимостей в контексте mesh-segmentation focus.

---

## Контекст

Под «листьями» понимаются модули `neurd/*.py` с низкой связностью
с ядром (`in/out ≤ 1` по числу внутренних импортов) — кандидаты на
независимый рефакторинг и тестирование. Полный список — в §7.1
[NEURD_STRUCTURE.md](NEURD_STRUCTURE.md).

Все правки делались **только в листьях и их зависимостях**; ядро
(`neuron.py`, `neuron_utils.py`, `preprocess_neuron.py`,
`proofreading_utils.py`, `error_detection.py`) **не трогалось** — там
циклические импорты и глобальное состояние через
`datasci_tools.module_utils`, нужны smoke-тесты.

---

## Инфраструктура тестов

**Текущее состояние (после Фазы 3, 2026-05-27): тесты прогоняются локально на
Python 3.12 + numpy 2 без Docker. `pytest tests/unit/` → 45 passed, 4 skipped.**

- Расположение: [tests/unit/](tests/unit/) — `leaves/`, `proximity/` и smoke-файлы
  Фазы 3 в корне `unit/` (`test_env_compat.py`, `test_numpy_compat.py`,
  `test_mesh_tools_compat.py`).
- Гард [tests/unit/__init__.py](tests/unit/__init__.py)
  `skip_if_datasci_tools_unusable()` теперь импортирует `neurd` **первым** —
  это активирует numpy-shim и ipyvolume-stub из
  [neurd/__init__.py](neurd/__init__.py) до пробы `datasci_tools.module_utils`.
  До Фазы 3 guard падал на `AttributeError: np.float_` и весь suite skipped.
- conftest = только `matplotlib.use("Agg")`.
- Установка локально: [scripts/install_local.sh](scripts/install_local.sh)
  (создаёт venv, ставит `mesh_processing_tools --no-deps` чтобы обойти пин
  `open3d==0.11.2`, потом [requirements-local.txt](requirements-local.txt)).
  Поддерживается Python 3.10–3.12 (3.13+ блокируется отсутствием open3d wheels).
- **Docker удалён** в Фазе 3 — старый `celiib/mesh_tools:v4` (Python 3.8)
  несовместим с новыми шимами; новый Dockerfile не делали, conda-env'а
  достаточно. Если когда-нибудь понадобится CI, базироваться на
  `python:3.12-slim` (рецепт из удалённого PHASE3_PLAN сохранён в git history).
- Skipped в текущем прогоне (объяснимо, не регрессия):
  - 2 теста `tests/unit/proximity/` — `pytest.importorskip("datajoint")`,
    datajoint опциональный (`extras_require[connectome]`).
  - 1 тест `test_ipyvolume_submodule_resolves_via_stub` — пропускается когда
    реальный `ipyvolume` установлен (его тянет `meshparty` как dep).
  - 1 — leftovers (см. `pytest -v`).
- Маппинг distribution-name → import-name (важно для setup.py / requirements):
  `datasci_stdlib_tools`→`datasci_tools`,
  `machine_learning_tools`→`machine_learning_tools`,
  `graph_nx_tools`→`graph_nx_tools`,
  `mesh_processing_tools`→`mesh_tools`,
  `neuron_morphology_tools`→`neuron_morphology_tools`.

---

## Что сделано по каждому листу

### [neurd/volume_utils.py](neurd/volume_utils.py) (100 LOC, без изменений по объёму)
- **Баг исправлен:** `nucleus_ids` → `nuclues_ids` (опечатка, ветка
  `return_centers=False` падала с `NameError`).
- **Тесты:** [test_volume_utils.py](tests/unit/leaves/test_volume_utils.py)
  — 5 кейсов (абстрактность ABC, init, set_synapse_filepath, дефолтные
  методы, регрессия на ту самую ветку).
- **Статус:** чистый. Дальше — формализовать как `Protocol` (для §7.3
  Volume-адаптеры), но это не приоритет.

### [neurd/functional_tuning_utils.py](neurd/functional_tuning_utils.py) — **удалён**
- Было 34 строки тонких обёрток над `nu.cdiff/cdist` + self-import,
  плюс `add_on_delta_to_df`, который через `ftu.cdist` вызывал
  самого себя.
- 2 вызова `ftu.add_on_delta_to_df` в [connectome_utils.py:2100, 2139](neurd/connectome_utils.py#L2100)
  заменены на прямые `nu.cdist(...)`.
- Из [connectome_utils.py](neurd/connectome_utils.py) убраны оба дублирующих
  `from . import functional_tuning_utils as ftu`.
- **Статус:** закрыто, возвращаться не к чему.

### [neurd/microns_graph_query_utils.py](neurd/microns_graph_query_utils.py) (152 → 113 LOC)
- Полный rewrite. Убран self-import (`from . import microns_graph_query_utils as mqu`),
  все внутренние `mqu.foo()` → прямые вызовы. Импорты `datasci_tools`
  вынесены наверх. Формат причёсан.
- Логика **не менялась** — тот же набор функций, та же сигнатура.
- **Тесты:** [test_microns_graph_query_utils.py](tests/unit/leaves/test_microns_graph_query_utils.py)
  — 9 кейсов на pandas-фильтры + патч лоадеров через `unittest.mock`.
- **Статус:** чистый.

### [neurd/nature_paper_plotting.py](neurd/nature_paper_plotting.py) (192 LOC, без изменений)
- Не трогал — это фигуры для статьи, риск сломать визуальный вывод.
- **Тесты:** [test_nature_paper_plotting.py](tests/unit/leaves/test_nature_paper_plotting.py)
  — только smoke (палитры + один `plot_edit_labels_subset` не падает).
  `matplotlib.use("Agg")` обязателен.
- **«Лист» с подвохом:** статический `from . import cell_type_utils`
  на верхнем уровне тянет цепочку
  `cell_type_utils → axon_utils → h01_volume_utils → microns_volume_utils → datajoint`.
  То есть фигуры из научной статьи требуют для импорта весь стек ядра
  + базу коннектома (`datajoint`). Это запах: модуль не самодостаточен,
  как должен бы быть для leaf. Тесты сейчас скипаются через
  `importorskip("datajoint")`.
- **Статус:** покрыт минимально. Кандидат на лёгкий рефакторинг:
  - вынести `from . import cell_type_utils` внутрь функций, где он
    реально нужен;
  - либо вынести палитры/чистые plot-функции в отдельный
    `neurd/plotting/figures.py` без зависимости от `cell_type_utils`.

### [neurd/parameter_utils.py](neurd/parameter_utils.py) (879 → **853 LOC**)
Основная работа первой сессии. Подробнее ниже.

---

## `parameter_utils.py` — детально

### Что починено

| ID | Что | Где |
|---|---|---|
| **B1** | `_clean_modules_dict`: правильная итерация по `{module: {att_type: {cat: {...}}}}` (раньше дважды итерировался `data`) | [строки ~328-340](neurd/parameter_utils.py#L328) |
| **B2** | `PackageParameters.__init__`: проверка на `data` (а не `filepath`), атрибут `_data` (а не `self.data`); shadowing в dict-comprehension убран | [строки ~180-200](neurd/parameter_utils.py#L180) |
| **B4** | `from os import sys` → `import sys`. Параллельно нормализован порядок импортов | [строки 1-15](neurd/parameter_utils.py#L1) |
| **B7** | Удалены `config_directory()` и `config_directory_name` (дубликат `parameter_config_folder(return_str=True)`) | [бывшие строки ~656-661] |
| **S2** | `modes_global_param_and_attributes_dict_from_module`: ветка `elif len(att_dicts)==1` ужата с 8 строк до 4, убраны мёртвое присваивание и `else: pass` | [~414-418](neurd/parameter_utils.py#L414) |
| **S4** | Удалены 14 строк закомментированных дублей `__getattr__`/`__setattr__` | [~59-72](neurd/parameter_utils.py#L59) |
| **S5** | 3 bare `except:` → `except KeyError` (×2) и `except (TypeError, ValueError)` (×1) | [~77, ~104, ~261](neurd/parameter_utils.py#L77) |
| **S8** | 4 сравнения через срез (`k[:2]=="__"`, `k[:N]==...`, `[-5:]!=".json"`) → `startswith`/`endswith` | разные места |
| **S11** | 8 импортов `datasci_tools` из подвала + `import copy` из середины вынесены наверх. Удалён неиспользуемый `import time` | [строки 1-15](neurd/parameter_utils.py#L1) |
| **A3** | Префикс `_` для 6 хелперов без внешних вызовов: `injest_nested_dict`, `jsonable_dict`, `clean_modules_dict`, `add_global_name_to_dict`, `parameter_dict_from_module_and_obj`, `this_directory` | весь файл |

Публичная поверхность сузилась с 18 функций до 11. После префиксации
ясно видно, что трогать `_*` безопасно (нет внешних callers — проверено
`grep`-ом по `*.py` и `*.ipynb` всего репо).

### Регрессионные тесты (новые)

В [test_parameter_utils.py](tests/unit/leaves/test_parameter_utils.py):
- `test_clean_modules_dict_strips_non_jsonable_in_leaf_dicts` — для B1.
- `test_package_parameters_can_be_constructed_from_another_instance` —
  для B2 (включая проверку `deepcopy`).
- `test_package_parameters_update_round_trip_uses_copy_branch` —
  упражняет ту же копи-ветку через публичный `.update()`.

Всего по `parameter_utils` — **14 unit-тестов**.

### Что в `parameter_utils.py` НЕ починено (по плану из ревью)

Я НЕ трогал следующие пункты из ревью — каждый имеет ненулевой риск:

| ID | Описание | Почему ждёт |
|---|---|---|
| **B3** | ~~`attr_map`: `key.replace(suf, "")` снимает суффикс ОТКУДА УГОДНО в строке~~ | **ИСПРАВЛЕНО** — `.replace(suf,"")` → `.removesuffix(suf)`. Тест `test_attr_map_does_not_strip_infix_global` добавлен. |
| **B5** | `parameters_from_filepath`: `exec(f"import {module_name}; from {module_name} import {dict_name}")` + `sys.path.append(directory)` навсегда. Плюс `filename.replace(".py","")` ломается для `"my.py.config.py"` | замена на `importlib.util.spec_from_file_location` меняет порядок side-effect'ов; нужны smoke-тесты на реальном `.py`-конфиге из [neurd/parameter_configs/](neurd/parameter_configs/) |
| **B6** | `set_parameters_for_directory_modules_from_obj`: shadowing переменной `k` между внешним и внутренним циклом; `exec(imp_str)` + `eval(k)` вместо `importlib.import_module`; цикл `for i in range(0,2)` — двойной импорт без комментария | это самая «горячая» функция модуля — она применяет конфиг к ВСЕМ модулям пакета (вызывается из `neurd.set_volume_params`). Любое изменение требует прогона интеграционного `tests/integration/test_autoproof_pipeline.py`, который сейчас не запускается локально |
| **S1** | ~~self-import `from . import parameter_utils as paru`~~ | **ИСПРАВЛЕНО** — убран, 5 `paru.X()` → прямые вызовы. |
| **S3** | `_injest_nested_dict(..., filter_away_suffixes=True)` — параметр принимается, но игнорируется | косметика, делать вместе с A1 |
| **S6** | `PackageParameters.module_attr_map`: при отсутствующем `module_name` молча возвращает `{}` — опечатка в имени модуля = тихая работа на дефолтах | желательно warning или strict-mode, но это изменение поведения; обсуждать |
| **S7** | `attr_map` возвращает `dict` или `list`, в зависимости от флагов — непоследовательный тип | надо обсудить желаемый контракт; либо всегда `dataclass`, либо namedtuple |
| **S9** | `PackageParameters.__setitem__` принимает что угодно (включая сырые dict'ы), хотя ожидается `Parameters` | простая корректировка: `self._data[k] = v if isinstance(v, Parameters) else Parameters(v)` — но **меняет поведение** в крайних случаях |
| **S10** | Коллизия `Parameters.dict` (property) с возможным ключом `"dict"` в данных | косметика — задокументировать |

### Архитектурные пункты (большая работа)

| ID | Описание |
|---|---|
| **A1** | `Parameters` совмещает 3 модели доступа (dict, attr, объект с `attr_map`) — переписать на `pydantic.BaseModel` или `dataclass`. Снимет 60% строк, добавит валидацию схемы. |
| **A2** | Конфигурирование через **мутацию атрибутов модулей** в `set_parameters_for_directory_modules_from_obj` — это главный механизм глобального состояния NEURD. Чтобы развязать ядро (§5 из NEURD_STRUCTURE.md), начинать придётся отсюда. |
| **A4** | Нет валидации схемы JSON-конфига. `pydantic` решает в 1 строку. |

---

## Блок Proximity (§7.2)

### Файлы
- [neurd/proximity_utils.py](neurd/proximity_utils.py): 944 → **920 LOC**.
- [neurd/proximity_analysis_utils.py](neurd/proximity_analysis_utils.py): 978 → **966 LOC**.

### Что сделано
- **Чистка импортов:** в обоих файлах импорты вынесены наверх; убраны дубли
  (`mvu`, `hvu`, `np`); удалены неиспользуемые `import datajoint as dj`
  (в `proximity_utils.py` — нигде не вызывался) и `module_utils as modu`
  (только в комментариях).
- **Self-imports убраны:** `from . import proximity_utils as pxu` и
  `from . import proximity_analysis_utils as pxa` удалены; внутренние
  `pxu.foo`/`pxa.foo` → прямые вызовы (7 + 1 правка).
- **Баг исправлен:** [proximity_analysis_utils.py:547](neurd/proximity_analysis_utils.py#L547)
  было `pxa.pairwis_df = pairwise_presyn_proximity_onto_postsyn(...)` —
  опечатка `pairwis` + присвоение **атрибута модуля** вместо локальной
  переменной. Никто не читал результат (опечатка делала его
  недоступным даже если бы и читали). Заменено на локальную `pairwise_df`.
- **Тесты:** новый каталог
  [tests/unit/proximity/](tests/unit/proximity/) с двумя файлами,
  всего **12 кейсов** на DB-free хелперы:
  - `proximity_utils`: `synapse_coordinates_from_df` (×2),
    `A_prox_from_G_prox`, `A_syn_from_G_prox` — итого 4;
  - `proximity_analysis_utils`: `conversion_rate` (×2), `conversion_df`
    (×2), `print_n_dict` (×2) — итого 6, плюс параметризованные
    варианты.
- **Гард переехал** в [tests/unit/__init__.py](tests/unit/__init__.py)
  (одно место для всего юнит-дерева). `tests/unit/leaves/__init__.py`
  теперь просто реэкспортирует.

### Чего НЕ хватает (для будущих заходов)
- **Большая часть `proximity_utils` и `proximity_analysis_utils`
  обращается к датабазе через глобальный `vdi`** — те же грабли A2 из
  `parameter_utils`. Без зеленого `tests/integration/` интеграцию
  не накрыть. После того как пользователь даст GO на запуск
  `test_autoproof_pipeline.py` в Docker — можно будет добавить
  smoke-тесты на DB-функции через моки `vdi`.
- `add_euclidean_dist_to_prox_df` (`pxa`) — частично pure, но
  использует `pu.append_df_to_source_target` /
  `pu.distance_between_coordinates` из `datasci_tools.pandas_utils`.
  Тестируется, но шумно — отложил.
- `proximity_pre_post`, `presyn_proximity_data`,
  `postsyn_proximity_data` — ядро модуля, целиком DB-bound.

### Запахи, замеченные, но не починены
- В обоих файлах используется глобальный `vdi`, не импортированный в
  модуль (устанавливается через `set_parameters_for_directory_modules_from_obj`).
  Это часть проблемы A2 — фундаментальная для NEURD. Не трогаем.
- Несколько закомментированных блоков по 5-20 строк, легко удаляются,
  но это для отдельного «cosmetic» прохода по всем модулям.

### Обнаружено в процессе: «скрытые тяжёлые импорты»
При прогоне в Docker (Python 3.8) `proximity_utils` подтягивал две
несовместимых цепочки на верхнем уровне:

1. **`mesh_tools.skeleton_utils` → meshparty → cloudvolume → fail на
   PEP 585.** Починено: импорт `sk` сделан **ленивым** —
   перенесён внутрь единственной использующей функции
   [proximity_pre_post](neurd/proximity_utils.py#L378). Теперь модуль
   импортируется без всей мешевой цепочки, и pure-helpers (`A_*`,
   `synapse_coordinates_from_df`) тестируются на голой системе.

2. **`from . import h01_volume_utils → microns_volume_utils → import
   datajoint`.** Не чинено: на верхнем уровне `microns_volume_utils.py`
   делает `import datajoint as dj`. Это значит, что **любой**
   NEURD-модуль, импортирующий `mvu`/`hvu`, требует `datajoint`. Это
   ещё один признак A2-связности. Решение — ленивизировать
   `import datajoint` в `microns_volume_utils`, но он широко
   используется как ядерный helper, и без интеграционных тестов
   трогать опасно. Пока обходим через `pytest.importorskip("datajoint")`
   в proximity-тестах.

   **Запомнить:** при следующих листьях (`motif_utils`, `gnn_*`) тоже
   делать `pytest.importorskip("datajoint")` на случай той же цепочки.

---

## Покрытие тестами

| Модуль | Тестов | Уровень |
|---|---|---|
| `volume_utils.py` | 5 | юнит, полный |
| `microns_graph_query_utils.py` | 9 | юнит, полный |
| `nature_paper_plotting.py` | 2 | smoke (skip без `datajoint`) |
| `parameter_utils.py` | 16 | юнит, частичный (+2 тест для B3) |
| `proximity_utils.py` | 4 | юнит, только DB-free часть |
| `proximity_analysis_utils.py` | 8 | юнит, только DB-free часть |
| **Итого** | **44** | |

Пробелы в `parameter_utils.py`:
- `attr_map` — частично покрыт (B3 исправлен и протестирован, 2 теста);
- `modes_global_param_and_attributes_dict_from_module` — НЕ покрыт
  (хотя B1 поправлен — `_clean_modules_dict` теперь работает корректно);
- `parameters_from_filepath` с реальным `.py`-конфигом — НЕ покрыт
  (B5 ждёт smoke-тест);
- `set_parameters_for_directory_modules_from_obj` — НЕ покрыт
  (B6 ждёт smoke-тест);
- `category_param_from_module` — НЕ покрыт (хотя реально используется
  снаружи: `soma_extraction_utils.py` и куча ноутбуков).

---

## Куда двигать систему дальше

### Шаг 1 (после того как Docker соберётся): зелёная база
1. `docker compose run --rm test` → убедиться, что все ~30 тестов
   зелёные.
2. Только после этого браться за изменения с риском поведения.

### Шаг 2 (низкий риск, можно делать сразу после Шага 1):
- ~~**S1**~~ — **сделано**.
- ~~**B3**~~ — **сделано** (`.replace` → `.removesuffix` + 2 новых теста).
- ~~**S3**~~ — **сделано** (удалён `filter_away_suffixes` из `_injest_nested_dict`).
- ~~Тесты на `modes_global_param_and_attributes_dict_from_module` и
  `category_param_from_module`~~ — **сделано** (9 тестов с фейковым модулем).

### Шаг 3 (средний риск):
- ~~**B5**~~ — **сделано** (`exec`/`eval` → `importlib.util.spec_from_file_location`,
  +6 тестов с фейковым `.py`-конфигом в `tmp_path`).
- ~~**B6 (часть `exec`/`eval`)**~~ — **сделано**
  (`exec("from pkg import mod")` → `importlib.import_module(f"pkg.mod")`,
  +2 end-to-end теста с фейк-пакетом на диске).
- **B6 (часть про двойной проход `for i in range(0, 2)`)** — **отложено намеренно**.
  Анализ (2026-05-27): второй проход с `plus_unused=True` фактически
  затирает первый (`setattr` перезаписывает; K1 ⊆ K2; ни `params_to_find`,
  ни `parameter_list_from_module` не зависят от того, что было настроено
  в первой итерации). Однопроходный эквивалент существует, но автор мог
  иметь в виду какой-то сценарий side-effect'ов через
  `set_parameters_for_directory_modules_from_obj` → `set_volume_params`,
  который без интеграционного теста (`test_autoproof_pipeline.py`) не
  верифицируется. Trigger для возврата: написать equivalence-тест с 3+
  модулями, имеющими взаимные `global_parameters_dict_*`-ссылки.

### Шаг 4 (другие листья §7.2):
- ~~**Proximity-block**~~ — **сделано** (self-imports, datajoint lazy, тесты).
- ~~**Self-imports в non-core модулях**~~ — **сделано** (сессия 2026-05-27):
  `branch_attr_utils`, `gnn_embedding_utils`, `neuron_geometry_utils`,
  `neuron_graph_lite_utils`, `motif_utils`, `graph_filters`,
  плюс ранее: `connectome_analysis_utils`, `connectome_query_utils`,
  `width_utils`, `soma_splitting_utils`, `neuron_simplification`.
- **GNN-block** (`gnn_embedding_utils`, `gnn_cell_typing_utils`, ~1.4k LOC) —
  вынести `torch`/`torch_geometric` в `extras_require["ml"]`, добавить юниты.
  Self-import уже убран; осталось только lazify torch-импорты.

### Шаг 5 (большая работа):
- **A1**: `Parameters` → `pydantic.BaseModel`. После этого открывается
  путь к §7.4 (распутывание ядра через формализацию конфига).
- **§7.3 Volume-адаптеры**: формализовать `VolumeDataInterface` как
  `Protocol`, обтестить vdi_* моками.

---

## Состояние git (на момент написания)

Все правки залиты в основную ветку (пользователь коммитит самостоятельно
по итогам каждой сессии). Ничего не отправлено в remote.

Изменённые/новые файлы по итогам всей работы с листьями:
```
M  neurd/connectome_utils.py            # убраны вызовы ftu
D  neurd/functional_tuning_utils.py     # удалён
M  neurd/microns_graph_query_utils.py   # rewrite (no logic change)
M  neurd/parameter_utils.py             # B1-B3, S1-S2, S4-S5, S8, S11, A3
M  neurd/volume_utils.py                # опечатка nucleus_ids

# Фаза 1+2 deps
M  requirements.txt                     # -7 пакетов, 13 base deps
M  setup.py                             # extras_require [connectome/viz/ml/all]
M  neurd/microns_volume_utils.py        # datajoint lazy; self-import убран
M  neurd/soma_extraction_utils.py       # datajoint import убран
M  neurd/proximity_analysis_utils.py    # datajoint lazy; seaborn soft
M  neurd/spine_utils.py                 # seaborn soft
M  neurd/nature_paper_plotting.py       # seaborn soft; cell_type_utils lazy; self-import убран
M  neurd/neuron_visualizations.py       # ipyvolume soft; self-import убран
M  neurd/h01_volume_utils.py            # unused imports убраны

# Self-imports убраны (сессии 2026-05-27)
M  neurd/branch_attr_utils.py
M  neurd/gnn_embedding_utils.py
M  neurd/graph_filters.py
M  neurd/motif_utils.py
M  neurd/neuron_geometry_utils.py
M  neurd/neuron_graph_lite_utils.py
M  neurd/connectome_analysis_utils.py
M  neurd/connectome_query_utils.py
M  neurd/neuron_simplification.py
M  neurd/soma_splitting_utils.py
M  neurd/width_utils.py

# Тесты
A  tests/unit/leaves/__init__.py
A  tests/unit/leaves/conftest.py
A  tests/unit/leaves/test_volume_utils.py
A  tests/unit/leaves/test_microns_graph_query_utils.py
A  tests/unit/leaves/test_nature_paper_plotting.py
A  tests/unit/leaves/test_parameter_utils.py      # 16 тестов, вкл. B3
A  tests/unit/proximity/test_proximity_utils.py
A  tests/unit/proximity/test_proximity_analysis_utils.py

# Фаза 3 (2026-05-27, ветка refactor_dependens)
M  neurd/__init__.py                    # numpy-shim + ipyvolume/cloudvolume stub-finder
M  tests/unit/__init__.py               # guard импортирует neurd первым
A  tests/unit/test_env_compat.py        # 5 тестов на шимы
A  tests/unit/test_numpy_compat.py      # 4 теста на numpy 2
A  tests/unit/test_mesh_tools_compat.py # 4 теста на mesh-стек
A  requirements-local.txt               # современные пины под Python 3.12
A  scripts/install_local.sh             # bootstrap venv 3.10–3.12
M  requirements.txt                     # numpy≥2, trimesh≥4, meshparty≥2, +h5py/tqdm/scikit-learn
M  README.md                            # секция Docker → Local install
D  docker/                              # вся папка удалена (~24 файла)
```

---

## Конвенции, которые я установил для этой работы

- **Не трогаем поведение без тестов.** B3/B5/B6/S6/S9 ждут, пока
  Docker даст возможность прогнать `test_autoproof_pipeline.py`.
- **Логические правки и косметика — разными коммитами.** Внутри одной
  сессии — да; в рамках одного PR — каждой группе свой коммит.
- **Регрессионный тест перед/одновременно с исправлением бага.**
  B1/B2 уже сделаны так. B3/B5/B6 надо так же.
- **Префикс `_` — после `grep`-проверки на внешних callers** (в `.py`
  и `.ipynb` отдельно — ноутбуки активно зовут публичный API).
- **Self-import (`from . import X as X`) убирать всегда** — это
  явный антипаттерн NEURD-кодстайла, который тащится во многих местах
  (был в `functional_tuning_utils`, `microns_graph_query_utils`, есть
  в `parameter_utils` и наверняка ещё где-то).
