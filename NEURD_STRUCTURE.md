# NEURD — Структура, зависимости и план дешёвой очистки

Этот файл — отправная точка для оптимизации форка NEURD.
Он описывает (1) внешние зависимости, (2) внутренние независимые блоки фреймворка,
(3) карту межмодульных связей и (4) места, где можно дёшево уменьшить кодовую базу
без серьёзного рефакторинга.

Всего в пакете `neurd/` ~94k строк Python в ~60 файлах.

---

## 1. Внешние зависимости

### 1.1. Указаны в `requirements.txt`
| Категория | Пакеты |
|---|---|
| Численные / научные | `numpy` (через `datasci_tools.numpy_dep`), `scipy`, `pandas`, `scikit-learn`, `pykdtree` |
| 3D-меши и скелеты | `trimesh==3.22.3`, `meshparty`, `pymeshfix`, `trimesh.ray.ray_pyembree` |
| Графы | `networkx` |
| Визуализация | `matplotlib`, `seaborn`, `ipyvolume` |
| Данные / БД | `datajoint`, `python-dotenv` |
| ML (опционально, закомментировано) | `torch`, `torch_geometric`, `pytorch_tools` |
| Внутренние пакеты автора | `datasci_tools`, `machine-learning-tools`, `graph-nx-tools`, `mesh_processing_tools` (`mesh_tools`), `neuron_morphology_tools`, `code_structure_tools`, `datasci-stdlib-tools` |

### 1.2. Что реально импортируется в коде (top-level)
`abc`, `collections`, `copy`, `dataclasses`, `datetime`, `functools`,
`importlib`, `os`, `pathlib`, `typing`, `inspect`, `operator`, `re`, `random`, `sys`, `time`,
`numpy` (как `datasci_tools.numpy_dep as np`), `pandas`, `scipy`, `sklearn`,
`networkx`, `trimesh`, `meshparty`, `mesh_tools`, `pykdtree`,
`matplotlib`, `seaborn`, `ipyvolume`,
`datajoint`, `dotenv`,
`torch`, `torch_geometric` (только в ML-модулях),
`datasci_tools.*` (≈25 подмодулей), `neuron_morphology_tools.*`.

`datasci_tools` — это «бог-пакет» автора, через который идёт почти всё (numpy,
pandas, matplotlib, networkx, ipyvolume, statistics, json, system_utils, и т.д.).

---

## 2. Логические блоки фреймворка

Модули можно сгруппировать по функциональным блокам.
В скобках — основные файлы и `LOC`.

### A. Ядро структуры нейрона (фундамент, всё опирается на него)
- [neuron.py](neurd/neuron.py) — 4316
- [neuron_utils.py](neurd/neuron_utils.py) — 10308 (самый импортируемый: 25+ других модулей)
- [branch_utils.py](neurd/branch_utils.py) — 1664
- [branch_attr_utils.py](neurd/branch_attr_utils.py) — 388
- [limb_utils.py](neurd/limb_utils.py) — 752
- [concept_network_utils.py](neurd/concept_network_utils.py) — 1839
- [neuron_graph_lite_utils.py](neurd/neuron_graph_lite_utils.py) — 956
- [neuron_geometry_utils.py](neurd/neuron_geometry_utils.py)
- [neuron_simplification.py](neurd/neuron_simplification.py) — 694
- [documentation_utils.py](neurd/documentation_utils.py)
- [parameter_utils.py](neurd/parameter_utils.py) — 879
- [width_utils.py](neurd/width_utils.py) — 548

### B. Препроцессинг и извлечение сомы
- [preprocess_neuron.py](neurd/preprocess_neuron.py) — 5409
- [soma_extraction_utils.py](neurd/soma_extraction_utils.py) — 2237
- [soma_splitting_utils.py](neurd/soma_splitting_utils.py) — 295
- [spine_utils.py](neurd/spine_utils.py) — 7063
- [synapse_utils.py](neurd/synapse_utils.py) — 4516

### C. Классификация / поиск / морфология
- [classification_utils.py](neurd/classification_utils.py) — 2830
- [cell_type_utils.py](neurd/cell_type_utils.py) — 1934
- [apical_utils.py](neurd/apical_utils.py) — 1822
- [axon_utils.py](neurd/axon_utils.py) — 3969
- [ais_utils.py](neurd/ais_utils.py) — 116 *(orphan)*
- [neuron_searching.py](neurd/neuron_searching.py) — 2376
- [neuron_statistics.py](neurd/neuron_statistics.py) — 4420

### D. Proofreading (детекция/исправление ошибок реконструкции)
- [proofreading_utils.py](neurd/proofreading_utils.py) — 8062
- [error_detection.py](neurd/error_detection.py) — 5326
- [graph_filters.py](neurd/graph_filters.py) — 1265
- [graph_filter_pipeline.py](neurd/graph_filter_pipeline.py) — 910 *(orphan)*
- [graph_error_detector.py](neurd/graph_error_detector.py) — 1364 *(orphan)*
- [graph_error_detector_dendrite.py](neurd/graph_error_detector_dendrite.py) — 931 *(orphan)*
- [graph_error_detector_axon.py](neurd/graph_error_detector_axon.py) — 163 *(orphan)*

### E. Коннектом / motif / proximity
- [connectome_utils.py](neurd/connectome_utils.py) — 2609
- [connectome_query_utils.py](neurd/connectome_query_utils.py)
- [connectome_analysis_utils.py](neurd/connectome_analysis_utils.py) — 499
- [motif_utils.py](neurd/motif_utils.py) — 1421
- [motif_null_utils.py](neurd/motif_null_utils.py) — **0 строк (пустой)**
- [proximity_utils.py](neurd/proximity_utils.py) — 944
- [proximity_analysis_utils.py](neurd/proximity_analysis_utils.py) — 978

### F. ML / GNN (опциональные — `torch_geometric`)
- [gnn_embedding_utils.py](neurd/gnn_embedding_utils.py) — 524
- [gnn_cell_typing_utils.py](neurd/gnn_cell_typing_utils.py) — 909 *(orphan)*

### G. Визуализация
- [neuron_visualizations.py](neurd/neuron_visualizations.py) — 4090
- [nature_paper_plotting.py](neurd/nature_paper_plotting.py)

### H. Volume / dataset adapters (microns vs h01)
- [microns_volume_utils.py](neurd/microns_volume_utils.py) — 699
- [microns_graph_query_utils.py](neurd/microns_graph_query_utils.py)
- [h01_volume_utils.py](neurd/h01_volume_utils.py) — 581
- [volume_utils.py](neurd/volume_utils.py) — 100
- [vdi_default.py](neurd/vdi_default.py) — 1208
- [vdi_microns.py](neurd/vdi_microns.py) — 64 *(orphan по импортам)*
- [vdi_microns_cave.py](neurd/vdi_microns_cave.py) — 131 *(orphan)*
- [vdi_h01.py](neurd/vdi_h01.py) — 105 *(orphan)*
- [cave_client_utils.py](neurd/cave_client_utils.py) — 517
- [cave_interface.py](neurd/cave_interface.py) — **1 строка реэкспорта**

### I. Пайплайн / IO / прочее
- [neuron_pipeline_utils.py](neurd/neuron_pipeline_utils.py) — 625 *(orphan, но используется тестами и ноутбуками)*
- [functional_tuning_utils.py](neurd/functional_tuning_utils.py)
- [nwb_utils.py](neurd/nwb_utils.py) — 80 *(orphan)*
- [dandi_utils.py](neurd/dandi_utils.py) — 15 *(orphan)*

### J. Параметры
- [parameter_configs/parameters_config_default.py](neurd/parameter_configs/parameters_config_default.py)
- [parameter_configs/parameters_config_h01.py](neurd/parameter_configs/parameters_config_h01.py)
- [parameter_configs/parameters_config_microns.py](neurd/parameter_configs/parameters_config_microns.py)
- `*_old.py` — устаревшие копии (см. §4)

### K. Legacy
- [legacy/preprocess_neuron.py](neurd/legacy/preprocess_neuron.py) — 2621
- [legacy/whole_neuron_classifier_datajoint_adapted.py](neurd/legacy/whole_neuron_classifier_datajoint_adapted.py) — 1405

---

## 3. Карта зависимостей (внутренние)

**Ядро (импортируется почти всеми):**
| Модуль | Сколько других файлов импортируют |
|---|---|
| `microns_volume_utils` | 28 |
| `h01_volume_utils` | 27 |
| `neuron_visualizations` | 26 |
| `neuron_utils` | 25 |
| `neuron_statistics` | 21 |
| `apical_utils` | 9 |
| `width_utils` | 8 |
| `cell_type_utils` | 7 |
| `error_detection`, `classification_utils` | 6 |
| `preprocess_neuron`, `neuron_simplification` | 5 |

**Полностью независимые («листья» — никто не импортирует):**
`ais_utils`, `cave_interface`, `dandi_utils`, `gnn_cell_typing_utils`,
`graph_error_detector_axon/dendrite/<core>`, `graph_filter_pipeline`,
`motif_null_utils`, `neuron_pipeline_utils`, `nwb_utils`,
`vdi_h01`, `vdi_microns`, `vdi_microns_cave`.

> Большая часть «листьев» — это либо точки входа (vdi_*, neuron_pipeline_utils),
> либо мёртвые/устаревшие модули. Это первый кандидат на удаление.

**Замечание про связность:** ядро (`neuron_utils`, `neuron`, `preprocess_neuron`,
`proofreading_utils`, `error_detection`) — клубок взаимных импортов, который
дёшево разорвать не получится. Любая «настоящая» оптимизация — это длинная работа.

---

## 4. Дешёвые удаления / очистки (минимальный риск)

Все пункты — это удаление либо тривиальная замена, без рефакторинга API.
Грубая оценка экономии — **~9–12 тыс. строк** + ~230 МБ артефактов.

### 4.1. Полностью мёртвые файлы (≈ 4 тыс. строк)
| Файл | LOC | Почему можно удалить |
|---|---|---|
| `neurd/motif_null_utils.py` | 0 | пустой |
| `neurd/cave_interface.py` | 1 | дублирующий `from .cave_client_utils import *` — заменить импортом напрямую |
| `neurd/dandi_utils.py` | 15 | никем не импортируется |
| `neurd/nwb_utils.py` | 80 | никем не импортируется |
| `neurd/ais_utils.py` | 116 | никем не импортируется (упоминается только в архивном ноутбуке) |
| `neurd/legacy/preprocess_neuron.py` | 2621 | папка `legacy/`, единственное упоминание — комментарий |
| `neurd/legacy/whole_neuron_classifier_datajoint_adapted.py` | 1405 | то же |
| `neurd/parameter_configs/parameters_config_default_old.py` | ≈900 | `_old`, ни на что не ссылается |
| `neurd/parameter_configs/parameters_config_h01_old.py` | ≈350 | то же |

Перед удалением `ais_utils`/`nwb_utils`/`dandi_utils` — `grep -r` по
`Applications/` и `docs/` (по моему просмотру они там только в .ipynb-выводе,
а не в реальных импортах).

### 4.2. Подсистема `graph_error_detector*` / `graph_filter_pipeline` (≈ 3.4 тыс. строк)
Файлы:
- `graph_error_detector.py` — 1364
- `graph_error_detector_dendrite.py` — 931
- `graph_error_detector_axon.py` — 163
- `graph_filter_pipeline.py` — 910

Импортируются только друг другом и одной ленивой строчкой в
`proofreading_utils.py:7205` (`from . import graph_filter_pipeline as pipe`
внутри функции). Похоже на параллельную/экспериментальную реализацию
proofreading-пайплайна, не подключённую к публичному API.

**Дёшево:** проверить, нужна ли эта одна точка входа в `proofreading_utils`;
если нет — удалить весь подграф (~3.4k LOC). Если нужна — оставить только её
и `graph_error_detector.py`, выкинуть dendrite/axon-варианты.

### 4.3. Функции-двойники `*_old` внутри живых файлов
~20 функций с суффиксом `_old`, которые сохранены параллельно с новыми
(`apical_shaft_classification_old`, `filter_axon_candiates_old`,
`complete_axon_processing_old`, `axon_width_like_segments_old`,
`high_degree_upstream_match_old`, `webbing_t_errors_limb_branch_dict_old`,
`high_degree_branch_errors_limb_branch_dict_old`,
`plot_branch_with_boutons_old`, `e_i_classification_from_neuron_obj_old`,
`split_suggestions_to_concept_networks_old`,
`axon_features_from_axon_sk_and_soma_center_old`, и др.).

Большинство не вызывается внутри `neurd/`. **Дёшево:** прогнать grep на каждую,
и удалять те, что не имеют внешних вызовов. Ожидаемая экономия — несколько
тысяч строк.

### 4.4. Дублирующиеся / лишние импорты
- `from pykdtree.kdtree import KDTree` встречается дважды подряд в
  `neuron_utils.py`, `neuron.py` (и ещё в 7 файлах одиночно).
- `from datasci_tools import module_utils as modu` встречается двойками
  (с разными пробельными хвостами).
- `from importlib import reload` импортирован в 4 файлах и нигде не нужен
  в продакшен-коде (отладочный артефакт).
- ~200+ строк закомментированного кода (`error_detection.py` — 53,
  `neuron_utils.py` — 29, `preprocess_neuron.py` — 28 и т.д.).

Это всё чистый dead-weight, безопасно убирается одной автоматической чисткой
(например `ruff --fix` + `autoflake --remove-unused-variables
--remove-all-unused-imports`).

### 4.5. Мусорные артефакты вне `neurd/`
- `temp/` — 21 МБ временных `.off`/`.mls` файлов от тестовых прогонов meshlab.
- `tests/` содержит 207 МБ — там лежат `*.off`/`*.pbz2` бинарники
  (`864691135510518224.pbz2`, `mesh_watertight.off`, и т.д.), а также
  каталоги `Poisson_temp/`, `PR_*`, `_tmp_*.tmp` — это рантайм-output, который
  должен быть в `.gitignore`, а не в репо.
- В корне `Applications/` лежит куча ноутбуков; для core-пакета их можно
  не трогать, но при сборке колеса исключить из `MANIFEST.in`.

**Дёшево:** добавить в `.gitignore` шаблоны (`temp/`, `tests/Poisson_temp/`,
`tests/PR_*`, `tests/_tmp_*`, `tests/*.off`, `tests/*.pbz2`), убрать файлы из
индекса `git rm --cached`. Это разгрузит ~230 МБ из git-истории (по факту
только из рабочей копии; для истории нужен `git filter-repo`, это уже не
дёшево).

### 4.6. `__init__.py`
Файл содержит закомментированный код (`load_all_modules_in_package`,
`set_volume_params()`). Чистка — секунда, риск нулевой.

### 4.7. `requirements.txt`
- Закомментированы `torch*` — но `gnn_*.py` всё равно требует их.
  Либо вернуть их как extras_require `[ml]`, либо честно вынести
  `gnn_*` в отдельный sub-package.
- `pymeshfix` указан в зависимостях, но ни одного `import pymeshfix` в `neurd/*.py`.
- `seaborn` используется только в `neuron_visualizations` / `nature_paper_plotting`.
  Можно перенести в extras `[viz]`.
- `datajoint` импортируется в нескольких местах для коннектома; если эта часть
  опциональна — тоже в extras `[connectome]`.

---

## 5. Что делать дальше (порядок по «дёшево → дорого»)

1. **Грязно-чистая уборка (полдня, нулевой риск):**
   §4.1, §4.4, §4.5, §4.6 — удаление пустых/мёртвых файлов, дубликатов
   импортов, артефактов из git.
2. **Удаление `_old`-функций (§4.3):** грепом по каждой, ~1–2 часа.
3. **Решение по `graph_error_detector*` (§4.2):** один разговор с автором/
   проверка тестов — затем удалить ~3.4k строк.
4. **Опциональные зависимости (§4.7):** разнести `torch/torch_geometric`,
   `seaborn`, `datajoint`, `dotenv` в `extras_require`, чтобы базовый
   `pip install neurd` ставил минимум.
5. **(уже не дёшево)** Распутать ядро `neuron_utils ↔ neuron ↔ preprocess_neuron ↔
   proofreading_utils ↔ error_detection`. Тут циклические импорты, общие
   глобальные параметры через `datasci_tools.module_utils`, и без тестов это
   опасно. Делать только после п.1–4 и после написания smoke-тестов
   (`tests/integration/test_autoproof_pipeline.py` — единственный, что есть).

---

## 6. Что уже сделано (безопасная очистка)

Удалено из репозитория:

| Файл | LOC | Причина |
|---|---|---|
| `neurd/motif_null_utils.py` | 0 | пустой |
| `neurd/cave_interface.py` | 1 | дублирующий реэкспорт |
| `neurd/dandi_utils.py` | 15 | никто не импортирует |
| `neurd/nwb_utils.py` | 80 | никто не импортирует |
| `neurd/ais_utils.py` | 116 | никто не импортирует |
| `neurd/legacy/` (вся папка) | 4026 | мёртвый код |
| `neurd/parameter_configs/parameters_config_default_old.py` | ~900 | `_old`, не ссылается |
| `neurd/parameter_configs/parameters_config_h01_old.py` | ~350 | то же |
| **Итого Python** | **≈ 5.5 тыс. строк** | |

Прочее:
- `neurd/__init__.py` — убраны мёртвые комментарии и закомментированные вызовы.
- `neurd/neuron.py`, `neurd/neuron_utils.py` — убраны дубликаты `from pykdtree.kdtree import KDTree`.
- `.gitignore` пополнен паттернами для рантайм-артефактов; `temp/` и
  ~80 файлов в `tests/` (Poisson_temp, PR_*, _tmp_*, *.off/*.mls в корне
  tests/) выведены из индекса (`git rm --cached`). `tests/fixtures/` (реальные
  фикстуры) и тестовый код сохранены.

Перед удалением каждый orphan был проверен `grep`-ом по всем `.py` —
ни одного реального импорта.

---

## 7. Модули, готовые к независимому рефакторингу/тестированию

Критерий — **низкая связность с ядром** (мало внутренних импортов в обе стороны)
и понятный изолированный домен. По возрастанию риска:

### 7.1. Полностью изолированные «листья» (out ≤ 1, in ≤ 1)
Можно вынести в отдельный subpackage и закрыть unit-тестами уже сейчас.

| Модуль | LOC | in | out | Домен |
|---|---|---|---|---|
| [neurd/volume_utils.py](neurd/volume_utils.py) | 100 | 4 | 0 | константы/утилиты по объёму |
| [neurd/parameter_utils.py](neurd/parameter_utils.py) | 879 | 3 | 1 | загрузка JSON-конфигов |
| [neurd/microns_graph_query_utils.py](neurd/microns_graph_query_utils.py) | — | 1 | 1 | dataset-specific запросы |
| [neurd/functional_tuning_utils.py](neurd/functional_tuning_utils.py) | — | 3 | 1 | direction-selectivity utils |
| [neurd/nature_paper_plotting.py](neurd/nature_paper_plotting.py) | — | 0 | 2 | конкретные фигуры статьи |

### 7.2. Тематически изолированные блоки (можно тестировать как «модуль из 2–3 файлов»)
Импортируются ядром, но **сами почти ни от чего не зависят**.

| Блок | Файлы | Что делает |
|---|---|---|
| **Proximity** | `proximity_utils.py` (944), `proximity_analysis_utils.py` (978) | геометрический proximity-анализ; обе стороны графа зависимостей ≈ 1 |
| **Motif** | `motif_utils.py` (1421) | мотивы в коннектоме; `motif_null_utils` уже выкинут |
| **GNN** | `gnn_embedding_utils.py` (524), `gnn_cell_typing_utils.py` (909) | единственные пользователи `torch_geometric`, ни одним другим модулем не используются — **можно вынести в `neurd.ml` extras** |
| **Connectome analysis** | `connectome_analysis_utils.py` (499), `connectome_query_utils.py` | надстройка над `connectome_utils`, тонкая |
| **Soma split** | `soma_splitting_utils.py` (295) | маленький модуль с понятным контрактом |
| **Width** | `width_utils.py` (548) | чистые функции вычисления ширины (in=8, но out=2 — pure compute) |

### 7.3. Volume-адаптеры (dataset-specific)
Файлы `vdi_microns.py`, `vdi_microns_cave.py`, `vdi_h01.py`, `vdi_default.py`,
`microns_volume_utils.py`, `h01_volume_utils.py`, `cave_client_utils.py` —
формально это plugin-слой над ядром. Их можно отрефакторить (привести к
единому интерфейсу `VolumeDataInterface`) **без затрагивания ядра**, если
зафиксировать публичный API.

### 7.4. Что **нельзя** трогать в одиночку (центральный клубок)
`neuron_utils`, `neuron`, `branch_utils`, `concept_network_utils`,
`preprocess_neuron`, `proofreading_utils`, `error_detection`,
`neuron_visualizations`, `neuron_statistics`, `axon_utils`, `apical_utils`,
`spine_utils`, `synapse_utils`, `classification_utils` — взаимные импорты,
общее глобальное состояние через `datasci_tools.module_utils`. Любая
переделка требует сначала smoke-тестов и плана разрыва циклов.

### Рекомендованный порядок «изолированной» работы
1. **GNN-block (§7.2)** — самый чистый кандидат: выкинуть `torch*` из core
   `requirements.txt` в `extras_require={"ml": [...]}`, накрыть юнит-тестами.
2. **Proximity-block (§7.2)** — добавить тесты на чистые геометрические функции.
3. **Volume-адаптеры (§7.3)** — формализовать `VolumeDataInterface` как
   `Protocol`/`ABC` и обтестить адаптеры моками.
4. **`parameter_utils` + `parameter_configs/` (§7.1)** — самый дешёвый
   rewrite: JSON-loader без бизнес-логики.
5. Только после этого — заходить в ядро.

