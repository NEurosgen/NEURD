# Рабочее состояние модулей и тестов

Документ для следующих сессий: какие модули в каком состоянии, что
покрыто тестами, какие задачи открыты.

См. также: [DEPS_PLAN.md](DEPS_PLAN.md) — лог рефакторинга зависимостей,
[NEURD_STRUCTURE.md](NEURD_STRUCTURE.md) — карта модулей.

---

## Инфраструктура тестов

**Прогон локальный, без Docker. Python 3.12 + numpy 2.**

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate neurd
pytest tests/unit/        # → 59 passed, 0 failed, 1 skipped
```

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
test_core_imports.py              11 тестов: импорт core-модулей после Фазы 5
                                  + PRE-1 assert (регресс-гард)
leaves/
  conftest.py                     matplotlib Agg
  test_parameter_utils.py         36 тестов
```
> Фаза 6: удалены `leaves/test_volume_utils.py` и `leaves/test_microns_graph_query_utils.py`
> (тестировали удалённые кластерные модули).

---

## Покрытие тестами

| Модуль | Тестов | Уровень |
|---|---|---|
| `parameter_utils.py` | 36 | юнит, расширенный |
| Core-модули (`test_core_imports.py`) | 11 | импорт-smoke, регресс-гард PRE-1 |
| Smoke-тесты Фазы 3 (env/numpy/mesh_tools) | 13 | шимы + версии |
| **Итого** | **60 passed, 1 skipped** | — |

### Пробелы
- **Нет функциональных тестов на pipeline** — нет характеризационного теста
  `test_segmentation_pipeline.py` на fixture-меше (был создан в Batch 1, не сохранился).
  Это главный пробел: нужен до замены meshlabserver и любых изменений в `Branch`/`Neuron`.
- Скипается 1 тест: `test_ipyvolume_submodule_resolves_via_stub` — реальный `ipyvolume`
  подтягивается через `meshparty`, stub-путь не активен.

---

## Состояние модулей

### Core-модули (чистые, Фаза 5 завершена)

| Модуль | LOC | Статус |
|---|---|---|
| `neuron.py` | 3455 | Очищен от syu/apu/au/clu/ssu/pru/nviz |
| `neuron_utils.py` | 9584 | Очищен от au/clu/apu/ed/pru/syu/nviz |
| `neuron_searching.py` | 2080 | Очищен от syu/au/clu/ed/nviz; ~35 мёртвых wrapper-функций удалено |
| `neuron_statistics.py` | 3727 | Очищен от apu/au/syu/ed/ipvu/mcu/mvu/hvu |
| `spine_utils.py` | 6442 | Очищен от syu/apu/ctu/nviz; bau-функции инлайнены |
| `branch_utils.py` | 1610 | Очищен от syu/au/bau |
| `concept_network_utils.py` | 1758 | Очищен от au/syu/nviz |
| `limb_utils.py` | 742 | Очищен от au/nst/nviz |
| `preprocess_neuron.py` | 5193 | Очищен от nviz |
| `parameter_utils.py` | 877 | Багфиксы B1–B7 + 36 тестов (без изменений в этой сессии) |
| `soma_extraction_utils.py` | 1786 | Без изменений в Фазе 5 |
| `width_utils.py` | 548 | Без изменений |

> **16 core-файлов** после Фаз 6–7 (+ `_cgal_segmentation`, `__init__`, `version`,
> `segmentation_pipeline`). `branch_attr_utils`/`documentation_utils` удалены (0 импортёров);
> `neuron_simplification` удалён в Фазе 7 (использовался только в урезанных стадиях 3–5).

### Удалены в Фазе 5
- `synapse_utils.py` (~4.5k LOC), `axon_utils.py` (~3.9k LOC),
  `apical_utils.py` (~1.8k LOC), `neuron_visualizations.py` (~4k LOC)

### Удалены в Фазе 6 (кластер целиком)
20 кластерных файлов (`proofreading_utils`, `error_detection`, `classification_utils`,
`cell_type_utils`, `graph_*`, `neuron_graph_lite_utils`, `neuron_pipeline_utils`,
`soma_splitting_utils`, `neuron_geometry_utils`, `vdi_*`, `volume_utils`,
`microns_*`, `h01_volume_utils`) + 2 мёртвых (`branch_attr_utils`, `documentation_utils`).
Детали и LOC — в [DEPS_PLAN.md](DEPS_PLAN.md) §Фаза 6.

---

## Открытые задачи

### Приоритет 1 — Тесты
- [ ] Заново создать характеризационный тест `tests/integration/test_segmentation_pipeline.py`
  БЕЗ зависимости от `vdi_*` (старый удалён в Фазе 6): fixture-меш →
  `segmentation_pipeline(mesh)` → проверить: somas≥1, limbs>0, у каждой ветки mesh+skeleton.
  Это контракт выхода slim-пути.

### Приоритет 2 — Удаление остатков кластера
- [x] **Готово (Фаза 6):** удалены все 20 кластерных файлов + 2 мёртвых. `neurd/` = 17 файлов.

### Приоритет 3 — Замена meshlabserver
Полностью убрать `meshlabserver`+`xvfb` = заменить 4 операции:
- [ ] Decimator → `open3d`/`trimesh` quadric decimation (легко)
- [ ] Poisson → `open3d` Screened Poisson (легко)
- [ ] FillHoles → `trimesh.fill_holes` (легко)
- [ ] Interior (Ambient Occlusion) → **нет прямого аналога** (сложно)
> Делать только после характеризационных тестов (Приоритет 1).

### Приоритет 4 — Оптимизация памяти
- [ ] `Branch.__init__`: `deepcopy` submesh для каждого Branch — главный драйвер RAM.
  Заменить на view/lazy — после характеризационных тестов.
- [ ] `deepcopy` целого Neuron на границах стадий (`return_copy=True`) — удваивает RAM.

---

## Конвенции, установленные в работе

- **Не трогаем поведение без тестов.** Особенно `Branch`/`Neuron`/`preprocess_neuron`.
- **Регрессионный тест одновременно с фиксом.**
- **Логические правки и косметика — разными коммитами.**
- **Self-import (`from . import X as X`) убирать всегда** — антипаттерн.
- **Не трогать upstream-пакеты** (`datasci_tools`, `mesh_tools`, ...).
- **Перед удалением функции — grep на живых вызывающих.**

---

## Известные смеллы (parameter_utils.py)

| ID | Описание | Почему ждёт |
|---|---|---|
| **B6 loop** | `for i in range(0, 2)` в `set_parameters_for_directory_modules_from_obj`. Второй проход затирает первый. | Без интеграционного теста на `set_volume_params` не верифицируется. |
| **S6** | `PackageParameters.module_attr_map`: при отсутствующем `module_name` молча возвращает `{}`. | Нужно решать: warning или strict-mode. |
| **S7** | `Parameters.attr_map` возвращает `dict` или `list` в зависимости от флагов. | Контракт обсудить с пользователем. |
