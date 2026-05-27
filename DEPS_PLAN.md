# План работы с зависимостями NEURD

Документ для будущих заходов: какие зависимости трогать, в каком порядке,
с каким риском. Цель — упростить стек **в контексте задачи сегментации меша
нейрона**, отказавшись от того, что нужно только для облачных пайплайнов
коннектома (MICrONS-style).

Дополняет [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md) и [LEAVES_WORK.md](LEAVES_WORK.md).

---

## 0. Контекст и принципы

### Глобальная цель
- **Поправить код ядра для упрощения сегментации меша нейрона.**
- Облачное скачивание данных (`cloudvolume`, `caveclient`, `datajoint`-сервер)
  **не в фокусе** — функционал можно потерять в обмен на меньшую сложность.
- ~~Анализ коннектома (проксимити, мотифы, GNN-классификация) — нужен в
  меньшей степени; держим работающим, но не оптимизируем под него.~~
  **Обновлено 2026-05-27:** connectome / proximity / motif / GNN **удалены
  целиком** как не относящиеся к сегментации меша. Туториалы в
  `Applications/Tutorials/{Proximities,GNN_*,Auto_Proof_Pipeline}/` могут
  ссылаться на удалённые модули — это принято.

### Принципы выбора
1. **Удаление > опциональность > замена > обновление.**
   В первую очередь убираем то, что не нужно. Только потом — рефакторим.
2. **Pure-pipeline всегда должен импортироваться.** Никакой тяжёлый
   опциональный модуль не должен тащить нас при импорте «листа».
3. **Никаких изменений без тестов на затронутом модуле.**
4. **Не трогать upstream-пакеты автора** (`datasci_tools`, `mesh_tools` и т.д.)
   — наш форк, у нас нет контроля над их PyPI-релизами. Работаем через
   обёртки/моки/локальные правки в `neurd/`.

---

## 1. Что выкидываем целиком

Эти зависимости можно убрать без потери функциональности, нужной для
сегментации меша.

| Пакет | Где упомянут | Почему можно убрать | Стоимость |
|---|---|---|---|
| `pymeshfix>=0.16.2` | `requirements.txt` | **Нигде не импортируется** в `neurd/*.py` (grep'нуто). Тянется транзитивно через `meshparty`. Из прямых зависимостей убирать безопасно. | 5 минут |
| `ipython_genutils` | `requirements.txt` | Deprecated с 2017. Не используется в коде. | 5 минут |
| `ipython` | `requirements.txt` | Нужен только для Jupyter-окружения, не для самого пакета. Перенести в notebooks-extras или вообще удалить из прямых. | 5 минут |
| `pykdtree>=1.3.7` | `proximity_*`, `neuron_utils`, `error_detection`, ещё ~3 модуля | Замена однострочная: `from scipy.spatial import KDTree`. Современный scipy на тех же бенчмарках сопоставим/быстрее. Минус один native wheel. | 1-2 часа + тесты |

**Эффект:** -4 прямых зависимости, **-1 native build dependency** (pykdtree
требует C-extension).

---

## 2. Что выносим в `extras_require`

Эти зависимости нужны для отдельных подсистем NEURD, но не для базового
mesh-pipeline. Должны быть опциональными.

| Зависимость | Подсистема | extras-имя |
|---|---|---|
| `datajoint>=0.12.9` | коннектом-таблицы, `proximity_*`, `connectome_*` | `[connectome]` |
| `seaborn>=0.12.2` | визуализация для статьи (`nature_paper_plotting.py`, `connectome_analysis`) | `[viz]` или `[paper]` |
| `torch`, `torch_geometric`, `pytorch_tools` | GNN-классификация (`gnn_*.py`) | `[ml]` |
| `dotmotif` (через git+url) | motif-анализ | `[motif]` |
| `tamarind` (через git+url) | motif-анализ | `[motif]` |
| `python-dotenv` | загрузка `.env` в `vdi_*` | `[connectome]` |
| `caveclient`, `cloud-volume` (сейчас в Dockerfile) | загрузка данных из облака MICrONS | `[cloud]` |
| `ipyvolume>=0.6.3` | интерактивная 3D-визуализация в Jupyter | `[viz]` |

Что должно остаться в **базовом** `requirements.txt`:
```
numpy, scipy, pandas, networkx, matplotlib, trimesh, meshparty
datasci_stdlib_tools, mesh_processing_tools, neuron_morphology_tools,
machine_learning_tools, graph_nx_tools, code_structure_tools
```

То есть только то, без чего mesh-pipeline не запустится.

**Стоимость:** ~1-2 часа на правку `setup.py` + ленивые импорты в местах,
где сейчас зависимости тянутся eagerly. **Минус 8 пакетов из обязательных**.

**Эффект:** базовый `pip install neurd` ставит ~10 пакетов вместо ~20+.
Чистая Docker-сборка становится быстрее в несколько раз.

---

## 3. Что заменяем на современное

| Что | На что | Где | Стоимость | Риск |
|---|---|---|---|---|
| `pykdtree.kdtree.KDTree` | `scipy.spatial.KDTree` | ~6 модулей (gre'p покажет) | 1-2 ч | низкий |
| `from os import sys` | `import sys` | уже сделано в `parameter_utils` | 0 | 0 |
| `k[:N] == ...` сравнения | `str.startswith` / `endswith` | уже сделано в `parameter_utils`; стоит пройтись по всем модулям | 30 мин | 0 |
| `exec(f"import {module_name}")` | `importlib.import_module` | `parameter_utils.parameters_from_filepath` и `set_parameters_for_directory_modules_from_obj` (см. B5/B6 в LEAVES_WORK.md) | 2-3 ч | средний — затрагивает `sys.path` |

---

## 4. Поднятие версий (выполнено в Фазе 3)

| Было | Стало | Заметка |
|---|---|---|
| `numpy<2` (через `datasci_tools.numpy_dep`) | **`numpy>=2,<3`** | Шим в `neurd/__init__.py` восстанавливает удалённые алиасы `float_`/`int_`/`complex_`. Не правим upstream. |
| `pandas>=2.0.3` | **`pandas>=2`** | Базовый pin, тестируется на 2.x в actual env. |
| `trimesh==3.22.3` (жёсткий пин!) | **`trimesh>=4`** | Современный mesh API; meshparty 2.0 совместим. |
| `meshparty>=1.16.13` | **`meshparty>=2.0`** | Проверено в env (`meshparty 2.x`). |
| Python 3.8 (Docker) | **Python 3.10–3.12 (local)** | install_local.sh форсит этот диапазон; 3.13+ блокируется отсутствием open3d wheels. |
| `cloudvolume` (top-level в `mesh_tools`) | — | Покрыт stub-finder'ом, реальный пакет не нужен для сегментации. |

---

## 5. Чего не трогаем

Это нужно держать на виду, чтобы случайно не врываться без плана.

| Пакет | Почему не трогаем |
|---|---|
| `meshparty` | Ядро mesh-pipeline. Allen-Institute, активно используется, заменять нечем без потери функциональности. **Кандидат на ленивизацию импорта, не на удаление.** |
| `mesh_tools` (`mesh_processing_tools`) | Пакет автора, обёртка над `meshparty`. То же самое. |
| `datasci_tools` | Главный источник хрупкости (см. §6 в LEAVES_WORK.md), но **трогать только постепенно, в рамках работы над каждым отдельным модулем**. Не делать «удалить целиком» одной задачей. |
| `networkx` | Современная, стабильная, ничего не сломано. |
| `scipy` | То же. |
| `matplotlib` | Базовый, используется почти везде, замена бессмысленна. |
| `neuron_morphology_tools` | Пакет автора, нужен для skeleton-операций ядра. |

---

## 6. Порядок выполнения

### ✅ Фаза 1 — Чистка очевидного (ВЫПОЛНЕНО)
1. ~~Удалить из `requirements.txt`: `pymeshfix`, `ipython_genutils`, `ipython`.~~ ✅
2. ~~Заменить `pykdtree.kdtree.KDTree` → `scipy.spatial.KDTree` во всех модулях.~~ ✅
3. ~~Удалить `pykdtree` из `requirements.txt`.~~ ✅

### ✅ Фаза 2 — Опциональность (ВЫПОЛНЕНО)
1. ~~Переписать `setup.py` с использованием `extras_require`.~~ ✅
2. ~~Разнести зависимости по группам `[connectome]`, `[viz]`, `[ml]`.~~ ✅ (`[motif]`/`[cloud]` — отложено, не критично)
3. ~~Ленивизировать импорты тяжёлых опциональных deps.~~ ✅
   - `datajoint`: 0 top-level imports (все lazy или удалены)
   - `seaborn`: 0 top-level imports (все soft)
   - `ipyvolume`: soft в `neuron_visualizations`
4. Документация: в `setup.py` есть комментарии `pip install neurd[connectome]` etc.

Дополнительно в рамках Фазы 2:
- Убраны self-imports во всех non-core модулях (20 осталось только в core clump).
- Баг B3 в `parameter_utils.attr_map` исправлен (`.replace` → `.removesuffix`).

### ✅ Фаза 3 — Освобождение от Docker и локальный запуск (ВЫПОЛНЕНО, 2026-05-27)

Цели достигнуты: тесты прогоняются локально на Python 3.12 + numpy 2 без Docker.

1. ✅ `neurd/__init__.py`: numpy shim (`np.float_` → `np.float64` для numpy 2)
2. ✅ `neurd/__init__.py`: meta-path stub-finder для `ipyvolume`/`cloudvolume`
   (покрывает submodule-импорты типа `from ipyvolume.moviemaker import MovieMaker`)
3. ✅ `scripts/install_local.sh` + `requirements-local.txt` для virtualenv 3.10–3.12
4. ✅ `requirements.txt` обновлён под numpy 2 / trimesh 4 / meshparty 2
5. ✅ Локальный `pytest tests/unit/` — **45 passed, 4 skipped**
6. ✅ Docker удалён целиком (вариант «сразу Шаг 5» из PHASE3_PLAN — пользователь
   работает в conda, Docker как CI-инструмент не нужен)

**Smoke-тесты Фазы 3:**
- [tests/unit/test_env_compat.py](tests/unit/test_env_compat.py) — 5 тестов на шимы
- [tests/unit/test_numpy_compat.py](tests/unit/test_numpy_compat.py) — 4 теста на numpy 2
- [tests/unit/test_mesh_tools_compat.py](tests/unit/test_mesh_tools_compat.py) — 4 теста на mesh-стек

**Side-effect:** [tests/unit/__init__.py](tests/unit/__init__.py) guard теперь
импортирует `neurd` первым (активирует шим), благодаря чему ранее skipped тесты
из `leaves/` (33 шт.) запускаются локально и зелёные.

Подробности и обоснования — в [PHASE3_PLAN.md](PHASE3_PLAN.md).

### Фаза 4 — Постепенный отказ от `datasci_tools` (растянуто)
Для каждого модуля, который мы трогаем в рамках leaves/proximity/motif/...:
- Заменять `nu.cdiff` → inline math (мы уже сделали в `connectome_utils`).
- Заменять `jsu.json_to_dict` → `json.load`.
- Заменять `xu.adjacency_matrix` → `nx.adjacency_matrix`.
- Заменять `gu.flatten_nested_dict` → локальный helper.
- Etc.

Не делать отдельной «удалить datasci_tools» задачей. Через 10-15 модулей
зависимость станет тонкой.

**Финальный шаг (когда возможно):** удалить `set_parameters_for_directory_modules_from_obj`
+ `modu.all_modules_set_global_parameters_and_attributes` (A2 в LEAVES_WORK.md).
Заменить на pydantic-`Parameters` + явное применение конфига. **Это
большой ход** — делается после того, как ядро стабилизировано тестами.

### Фаза 5 — Замена `ipyvolume` на `pyvista` (опционально, недели работы)
Делается **только если** будем активно работать с визуализацией. Иначе
старый `ipyvolume` будет работать ещё годы. Польза — современный VTK-стек,
лучшая интеграция с jupyter/web.

---

## 7. Конкретные действия для Фазы 1 (готовое к выполнению)

### Чистка requirements.txt
```diff
- pymeshfix>=0.16.2
- pykdtree>=1.3.7
  ipython
- ipython_genutils
```

### Замена `pykdtree` → `scipy.spatial.KDTree`
Grep-инвентаризация:
```bash
grep -rn "pykdtree.kdtree" --include="*.py" neurd/
```

На момент написания: использовалось в `proximity_utils.py`,
`proximity_analysis_utils.py`, `neuron_utils.py`, `preprocess_neuron.py`,
`error_detection.py`, `neuron_graph_lite_utils.py`, `soma_extraction_utils.py`,
`spine_utils.py`, `neuron.py`. ~10 файлов.

Замена sed-style:
```bash
# во всех файлах ../neurd/*.py:
# from pykdtree.kdtree import KDTree → from scipy.spatial import KDTree
```

**Совместимость API:** в обоих API `KDTree(points).query(query_points)`
работает идентично для базового случая. Различия: pykdtree возвращает
`(dists, idx)`; scipy тоже `(dists, idx)`. Гранулярных различий нет в
типовых паттернах NEURD — все вызовы `KDTree(coords).query(other_coords)`
или `query_pairs(radius)` совместимы.

**Тесты:** на каждый модуль, который трогаем — smoke-тест что
`KDTree(np.random.rand(100,3)).query(np.random.rand(10,3))` возвращает
ожидаемые формы. Если есть существующие тесты на этих модулях
(`proximity_utils` уже имеет 4 теста) — прогнать после замены.

---

## 8. Метрики «до/после»

| Метрика | Базовое | После Фазы 1 ✅ | После Фазы 2 ✅ | После Фазы 3 ✅ |
|---|---|---|---|---|
| Прямых deps в requirements.txt | 17 | 13 | 13 | **16** (+ numpy/h5py/tqdm явно) |
| Опциональных групп | 0 | 0 | 4 [connectome/viz/ml/all] | 4 |
| top-level `datajoint` imports | ~5 | ~5 | 0 | 0 |
| top-level `seaborn` imports | ~3 | ~3 | 0 (soft) | 0 |
| Self-imports в non-core модулях | 65 | 65 | 20 (core only) | 20 |
| Native wheels (требуют компиляции) | ~3 | ~2 | ~2 | ~2 |
| Python version | 3.8 (Docker) | 3.8 | 3.8 | **3.10–3.12 (local)** |
| numpy | <2 | <2 | <2 | **≥2** |
| trimesh | ==3.22.3 | ==3.22.3 | ==3.22.3 | **≥4** |
| Зависимость от `celiib/mesh_tools:v4` | ✅ обязательна | ✅ | ✅ | **❌ удалена** |
| Локальный `pytest tests/unit/` | ❌ всё skip | ❌ skip | ❌ skip | **✅ 45 passed** |
