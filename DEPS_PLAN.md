# План работы с зависимостями NEURD

Документ для следующих сессий: какие зависимости трогать, в каком порядке,
с каким риском. Цель — упростить стек **в контексте задачи сегментации меша
нейрона**, отказавшись от всего, что нужно только для post-сегментационного
анализа.

См. также: [LEAVES_WORK.md](LEAVES_WORK.md) — рабочее состояние модулей и
тестов.

---

## 0. Контекст и принципы

### Глобальная цель
**Оставить в фреймворке только то, что нужно для сегментации меша нейрона.**

- Облачное скачивание данных (`cloudvolume`, `caveclient`, `datajoint`-сервер)
  — **не в фокусе**, функционал можно потерять в обмен на меньшую сложность.
- Post-сегментационный анализ (connectome, motif, proximity, GNN-классификация)
  — **удалён целиком в 2026-05-27**.

### Принципы
1. **Удаление > опциональность > замена > обновление.**
   Сначала убираем то, что не нужно. Только потом рефакторим.
2. **Pure-pipeline всегда должен импортироваться** без cloud/viz/ML-пакетов.
3. **Никаких изменений без тестов на затронутом модуле.**
4. **Не трогать upstream-пакеты автора** (`datasci_tools`, `mesh_tools`,
   `meshparty`, `neuron_morphology_tools`). У нас нет контроля над PyPI-релизами;
   работаем через шимы/обёртки в `neurd/`.
5. **Core clump не трогать без интеграционных тестов.** Это `neuron_utils`,
   `proofreading_utils`, `error_detection`, `spine_utils`, `axon_utils`,
   `apical_utils`, `preprocess_neuron`, `neuron_visualizations`,
   `soma_extraction_utils`, `concept_network_utils`, `branch_utils`,
   `limb_utils`, `neuron_searching`, `neuron_statistics`, `classification_utils`,
   `cell_type_utils`, `synapse_utils`.

---

## 1. Что уже сделано (Фазы 1–4)

### Фаза 1 — базовая чистка ✅
- Из `requirements.txt` убраны `pymeshfix`, `ipython`, `ipython_genutils`, `pykdtree`.
- `pykdtree.KDTree` → `scipy.spatial.KDTree` в 12 модулях.

### Фаза 2 — опциональность ✅
- `setup.py`: `extras_require = {connectome, viz, all}`.
- Top-level импортов `datajoint`/`seaborn` нет нигде.
- `ipyvolume` — soft import.
- Self-imports убраны во всех non-core модулях.

### Фаза 3 — Docker → локальный Python 3.12 ✅
- `neurd/__init__.py`: numpy-shim (`float_`/`int_`/`complex_` для numpy 2)
  + meta-path stub-finder для `ipyvolume`/`cloudvolume`
  (покрывает submodule-импорты).
- `requirements.txt`: numpy≥2, trimesh≥4, meshparty≥2, +h5py/tqdm/scikit-learn.
- `requirements-local.txt` + `scripts/install_local.sh` для venv 3.10–3.12.
- README: секция Docker → Local install.
- Папка `docker/` удалена целиком.
- `tests/unit/__init__.py` guard импортирует `neurd` первым (активирует шим).

### Фаза 4 — массовые удаления non-segmentation ✅
- **GNN** удалён: `gnn_embedding_utils.py` (524), `gnn_cell_typing_utils.py` (909),
  `Applications/Tutorials/GNN_*/`, `extras_require['ml']`.
- **Connectome** удалён: `connectome_utils.py` (2606), `connectome_analysis_utils.py`
  (499), `connectome_query_utils.py` (230).
- **Motif** удалён: `motif_utils.py` (1420).
- **Proximity** удалён: `proximity_utils.py` (926), `proximity_analysis_utils.py`
  (969), а с ними и `tests/unit/proximity/` (~400 LOC).
- `functional_tuning_utils.py` удалён ранее (тонкие обёртки над `nu.cdist`).
- Orphan-секции в `parameter_configs/parameters_config_{default,h01}.py` вычищены.

**Суммарно убрано из репо: ~16k LOC прикладного кода + ~400 LOC тестов.**

### parameter_utils.py — детальный рефакторинг ✅
- B1/B2/B4/B7: баги фикс + регресс-тесты.
- B3: `.replace(suf,"")` → `.removesuffix(suf)` (не снимал суффикс из середины).
- B5/B6: `exec`/`eval` → `importlib.util.spec_from_file_location` и
  `importlib.import_module`. +8 тестов с фейковым `.py`-конфигом / фейк-пакетом.
- S1/S3/S4/S5/S8/S11: косметика + self-import.

**Известный отложенный smell в B6:** двойной проход `for i in range(0, 2)` —
второй проход с `plus_unused=True` затирает первый. По анализу однопроходный
эквивалент работает идентично, но без интеграционного теста на
`set_volume_params`-цепочку лучше не трогать.

---

## 2. Текущее состояние стека

### Базовые зависимости (`requirements.txt`)
```
numpy>=2,<3       scipy           pandas>=2       networkx>=3
matplotlib>=3.7   h5py            tqdm            scikit-learn>=1.3
trimesh>=4        meshparty>=2.0

# Авторские (форк-зависимые)
datasci-stdlib-tools     machine-learning-tools     graph-nx-tools
mesh_processing_tools    neuron_morphology_tools    code_structure_tools
```

### `extras_require`
- `[connectome]` — `datajoint`, `python-dotenv` (для опциональных `vdi_*` адаптеров)
- `[viz]` — `seaborn`, `ipyvolume`
- `[all]` — всё вышеперечисленное

### Локальный запуск
- Python **3.10–3.12** (3.13+ блокирован отсутствием open3d-wheels)
- `bash scripts/install_local.sh` создаёт venv и ставит всё
- `pytest tests/unit/` → 62 passed, 2 skipped (под Python 3.12, numpy 2.4.6)

---

## 3. Что осталось не-сегментационного, но всё ещё в репо

| Модуль | LOC | Связь с ядром | Что с ним делать |
|---|---|---|---|
| `cell_type_utils.py` | 1934 | **7 импортёров вкл. `proofreading_utils`, `spine_utils`** | Сидит в core clump. Удалять нельзя без работы с ядром. Рассмотреть после Phase 5. |
| ~~`cave_client_utils.py`~~ | удалён | — | — |
| ~~`nature_paper_plotting.py`~~ | удалён | — | — |
| ~~`vdi_microns_cave.py`~~ | удалён | — | — |
| `microns_volume_utils.py`, `h01_volume_utils.py` | small | Адаптеры под конкретные датасеты MICrONS / H01 | Часть пользовательского API. Оставляем. |

---

## 4. Известные pre-existing проблемы

Эти баги обнаружены, но **не введены** рефакторингом — присутствовали до
ветки. Документируем для будущих сессий.

| ID | Где | Симптом | Причина | Статус |
|---|---|---|---|---|
| **PRE-1** | `neuron_searching.py:2306` | `NameError: 'fcu' is not defined` при `from neurd import spine_utils` на свежем интерпретаторе | `fcu = function_utils` импортировался на строке 2370 (ниже использования на 2306). Use-before-import. | ✅ Исправлено (Шаг A) — импорт поднят в шапку модуля. |
| **PRE-2** | `datasci_tools/dj_utils.py:13` (upstream) | `TypeError: 'ModuleNotFoundError' object is not callable` когда `datajoint` не установлен | `raise e("Datajoint must be installed...")` — `e` это caught instance, не класс. Upstream-баг. | ✅ Обойдён — `dj_utils` лениво импортируется в `spine_utils.py` только в `dj`-ветках. Сам upstream-баг остаётся. |
| **PRE-3** | `datasci_tools/dotmotif_utils.py:73` (upstream) | `TypeError: 'ModuleNotFoundError' object is not callable` когда `dotmotif` не установлен | Тот же паттерн `raise e(...)`. Вскрылся после фикса PRE-1. | ✅ Обойдён — `dotmotif_utils` лениво импортируется в `graph_filters.py` внутри `graph_filter_adapter`. |

**PRE-1 был главным блокером** для расцепления core. Теперь ядро импортируется
на голой системе (без `datajoint`/`dotmotif`/`caveclient`) — путь к smoke-тестам
ядра открыт.

---

## 5. Чего НЕ трогаем

| Пакет/модуль | Почему |
|---|---|
| `meshparty` | Ядро mesh-pipeline (Allen Institute). Замены нет. |
| `mesh_processing_tools` (= `mesh_tools`) | Авторская обёртка над meshparty. То же. |
| `neuron_morphology_tools` | Skeleton-операции ядра. |
| `networkx`, `scipy`, `matplotlib`, `trimesh` | Современные, стабильные. |
| `datasci_tools` | Большой источник хрупкости, **но трогать только в рамках конкретного модуля**, не «удалить целиком» одной задачей. См. §6 ниже. |
| Core clump | Список в §0. Нужны интеграционные тесты. |

---

## 6. Что осталось сделать

### Шаг A — починить PRE-1 (низкий риск, высокая ценность) ✅
В `neuron_searching.py` `from datasci_tools import function_utils as fcu` поднят
со строки 2370 в шапку модуля. Smoke-тест
`python -c "import neurd; from neurd import spine_utils"` → `OK`, без traceback.

Фикс PRE-1 вскрыл ещё два upstream `raise e(...)`-бага в той же цепочке импорта
(PRE-2 `dj_utils`, PRE-3 `dotmotif_utils`), раньше замаскированных падением на `fcu`.
Обойдены ленивыми импортами в местах единственного использования:
- `graph_filters.py`: `dotmotif_utils as dmu` → внутрь `graph_filter_adapter` (~336).
- `spine_utils.py`: `dj_utils as dju` → в обе ветки `if table_type == "dj":` (174, 186).

После всех трёх: `pytest tests/unit/` → 62 passed, 1 skipped (было 2 skipped — один
guard на доступность `datasci_tools` теперь проходит, т.к. ядро импортируется чисто).

**Польза:** открыта дорога к smoke-тестам ядра — теперь можно браться за Шаг C/D.

**Core smoke-тесты добавлены** (`tests/unit/test_core_imports.py`): параметризованный
импорт всех 17 модулей core clump + `graph_filters`, плюс точечная проверка PRE-1
(`neuron_searching.fcu` на module level). 19 тестов — регресс-гард на голой системе.
`pytest tests/unit/` → 81 passed, 1 skipped.

### Шаг B — точечные удаления non-segmentation ✅
`cave_client_utils.py`, `vdi_microns_cave.py`, `nature_paper_plotting.py` удалены.
Lazy-импорт `nature_paper_plotting` в `spine_utils.py:6692` был уже закомментирован.
`cave_client_utils` упомянут в docstring `vdi_default.py:398` — безопасно, не импорт.

### Шаг C — расцепить cell_type_utils от ядра ⏸️ ОТЛОЖЕНО (переоценено 2026-05-28)
Анализ показал, что план был основан на неверной предпосылке («7 импортёров»):
- Реальных внешних импортёров **3**: `proofreading_utils`, `neuron_pipeline_utils`,
  `spine_utils`. Поверхность — 6 call-sites, 2 различные сущности.
- `cell_type_utils` тянет только лёгкое (matplotlib/pandas/sklearn) → **не блокер
  импорта**. Lazy-import ничего не разблокирует.
- `spine_utils` (3 места) — лишь `hue_order` в plotting-функции (косметика).
- `proofreading_utils`/`neuron_pipeline_utils` — `e_i_classification_from_neuron_obj`
  как **шаг конвейера**, чей результат питает классификацию аксона
  (`inh_exc_class_to_use_for_axon`). Это настоящая связанность, а не «анализ сбоку».

**Вывод:** удалять нельзя без изменения поведения pipeline; lazy-import не даёт пользы.
Откладываем до тех пор, пока не появятся core smoke-тесты и не будет решено, выносить
ли E/I-классификацию из конвейера осознанно.

### Шаг D — Phase 5: `Parameters` → pydantic (большая работа)
- `parameter_utils.Parameters/PackageParameters` → `pydantic.BaseModel`.
- Это убирает `datasci_tools.module_utils` (27 call-sites, A2-проблема) —
  главный механизм глобального состояния NEURD.
- После: можно браться за `set_volume_params` и расцеплять core clump.
- **Требует** PRE-1 фикса + smoke-тестов на ядре.

### Шаг E — точечная Phase 4 (низкий приоритет)
- `jsu = json_utils` (2 call-sites) → stdlib `json`.
- `dsu.DictType` (1 call-site в `parameter_utils._jsonable_dict`) → inline check.
- Эти замены дают символическую пользу — `datasci_tools` остаётся в зависимостях
  пока есть хоть один импорт. Делать только мимоходом при работе с конкретным
  модулем.

---

## 7. Метрики

| Метрика | До | Сейчас |
|---|---|---|
| Python version | 3.8 (Docker) | **3.10–3.12 (local)** |
| Прямых deps в requirements.txt | 17 | 16 (явные, включая numpy/h5py/tqdm) |
| Опциональных групп | 0 | 3 [connectome/viz/all] |
| numpy | <2 | **≥2** |
| trimesh | ==3.22.3 | **≥4** |
| meshparty | >=1.16.13 | **≥2.0** |
| Top-level `datajoint`/`seaborn` imports | ~8 | **0** |
| Self-imports вне core | ~45 | **0** |
| Self-imports в core clump | 20 | 17 (часть удалена с модулями) |
| `celiib/mesh_tools:v4` Docker dep | ✅ обязательна | **❌ удалена** |
| Локальный `pytest tests/unit/` | ❌ skip-all | **✅ 81 passed, 1 skipped** (вкл. 19 core-import smoke) |
| Файлов в `neurd/*.py` | ~60 | **44** |
| Прикладного кода удалено | — | **~16k LOC** |
