# NEURD — план дальнейшей чистки (актуально 2026-05-30)

Состояние: `neurd/` = **16 файлов, ~24.8k строк**. Slim-пайплайн `mesh → Neuron`
зелёный (unit 59 passed / 1 skipped; характеризационный тест 4 passed, ~11 мин).
Zero-ref dead-code исчерпан (см. [DEAD_CODE_REACHABILITY.md](DEAD_CODE_REACHABILITY.md)).

Этот файл — приоритизированный план «как чистить дальше», с опорой на фактический
граф импортов (построен скриптом, не на глаз). Каждый шаг: что, зачем, риск, как проверять.

---

## 1. Фактический граф внутренних импортов (источник всех решений ниже)

```
branch_utils          -> neuron_statistics, neuron_utils, spine_utils, width_utils
concept_network_utils -> neuron_statistics, neuron_utils, width_utils
limb_utils            -> branch_utils, concept_network_utils, neuron_searching, neuron_statistics, neuron_utils
neuron                -> branch_utils, neuron_statistics, neuron_utils, preprocess_neuron, soma_extraction_utils, spine_utils, width_utils
neuron_searching      -> branch_utils, concept_network_utils, neuron_statistics, neuron_utils, width_utils
neuron_statistics     -> branch_utils, concept_network_utils, neuron_searching, neuron_utils
neuron_utils          -> neuron                       # ← god-hub, и при этом часть цикла
preprocess_neuron     -> neuron, neuron_utils, soma_extraction_utils
spine_utils           -> branch_utils, neuron_searching, neuron_statistics, neuron_utils, width_utils
width_utils           -> neuron_utils
soma_extraction_utils -> parameter_utils
segmentation_pipeline -> neuron, soma_extraction_utils
parameter_utils       -> (none)
```

**6 двусторонних циклов** (держатся на bottom-of-file импортах — band-aid):
- `neuron_utils ↔ neuron`  ← КОРНЕВОЙ, держится на **1 строке**
- `preprocess_neuron ↔ neuron`
- `neuron_statistics ↔ neuron_searching`
- `neuron_statistics ↔ concept_network_utils`
- `neuron_statistics ↔ branch_utils`
- `branch_utils ↔ spine_utils`

**7 self-import band-aid'ов** (`from . import X as X` в конце файла, антипаттерн по
конвенции проекта): soma_extraction_utils, concept_network_utils, branch_utils,
limb_utils, neuron_searching, preprocess_neuron, spine_utils.

Размеры (LOC): preprocess_neuron 4842, neuron_utils 3506, spine_utils 3419,
neuron 3402, neuron_statistics 1994, soma_extraction_utils 1828, neuron_searching 1816,
concept_network_utils 1341, branch_utils 881, parameter_utils 877, width_utils 457,
limb_utils 122.

---

## 2. Приоритеты (от дешёвого/безопасного к дорогому/рискованному)

### P1 — разорвать цикл `neuron_utils ↔ neuron` (дёшево, высокая отдача) ⭐
`neuron_utils` — god-hub: его импортируют 8 модулей. Но сам он лезет в `neuron`
**ровно один раз**: `neuron.Branch(starting_edge).endpoints` в
[neuron_utils.py:426](neurd/neuron_utils.py#L426). Цикл держится на этой строке +
band-aid `from . import neuron` (низ файла).

**Действие:** убрать модульный `from . import neuron`; внутри той функции сделать
локальный `from neurd.neuron import Branch` (или передавать endpoints аргументом).
Тогда `neuron_utils` грузится автономно, и большой кусок графа выпрямляется.

**Риск:** низкий. **Проверка:** fast-gate (`import neurd`, `pytest tests/unit/`) +
характеризационный тест (Branch строится на пути декомпозиции).

### P2 — убрать 7 self-import'ов (дёшево, конвенция проекта)
В каждом из 7 модулей в конце файла висит `from . import <self> as <alias>`, а тело
вызывает `alias.foo()` вместо `foo()`. По конвенции их надо убрать: заменить
`alias.` → прямой вызов, удалить импорт.

**Риск:** низкий, но механический объём (грепом по каждому alias). Делать **по одному
модулю на коммит** (логические правки отдельно от косметики — конвенция).
**Проверка:** fast-gate на каждый; характеризационный тест на батч.

### P3 — выделить low-level helpers из god-hub `neuron_utils` (среднее)
После P1 `neuron_utils` всё ещё 3.5k и тянется всеми. Идея пользователя «отдельный
файл с часто используемыми функциями» здесь работает: вынести самые
переиспользуемые чистые хелперы (не требующие классов Neuron/Limb/Branch) в новый
модуль без внутрипакетных зависимостей (`neuron_base_utils.py` / `_base.py`).
Тогда branch_utils/concept_network_utils/width_utils/… импортируют их из «листа»
графа, а не из god-hub → меньше рёбер к `neuron_utils`.

**Как выбирать что выносить:** функции из `neuron_utils`, которые (а) вызываются ≥3
модулями и (б) не используют `neuron.*`/класс-объекты. Список собрать грепом
`nru.<name>` по `neurd/`.
**Риск:** средний (широкие callers). **Проверка:** fast-gate + характеризационный.

### P4 — разбить два самых больших файла на пакеты (структурно, без выигрыша LOC)
Только ради читаемости/навигации; импорты это НЕ упрощает само по себе.

- **neuron.py (3402, идея пользователя)** → пакет `neuron/` c
  `_branch.py` (Branch 90–666), `_limb.py` (Limb 667–1890), `_soma.py` (Soma 1891–2068),
  `_core.py` (Neuron 2069–3402) и `__init__.py`, который реэкспортит все 4 класса
  (`from ._branch import Branch` …) → внешний API `from neurd.neuron import Neuron`
  не ломается. **Важно:** классы ссылаются друг на друга (Neuron→Limb→Branch) и на
  `nru` (который тянет Branch назад) — порядок реэкспорта и локальные импорты внутри
  классов критичны. **Делать только ПОСЛЕ P1** (иначе цикл размажется по 4 файлам и
  станет хуже). **Риск:** выше среднего.
- **preprocess_neuron.py (4842)** — самый большой; разбить по фазам препроцесса в
  пакет аналогично. Сначала прогнать на нём мёртвый код (см. P5).

**Проверка:** обязательно характеризационный тест (оба файла — ядро декомпозиции).

### P5 — глубокий dead-code (mark-and-sweep), как описано в DEAD_CODE_REACHABILITY.md
Zero-ref исчерпан, но взаимно-ссылающиеся мёртвые кластеры (A↔B, оба недостижимы)
им не ловятся. Метод (граф достижимости от корней-точек входа, over-approx динамики)
расписан в [DEAD_CODE_REACHABILITY.md](DEAD_CODE_REACHABILITY.md) §2–§5. Самый
вероятный улов — `calculate_decomposition_products` и его stats-цепочка (вне slim-пути).

**Риск:** средний/высокий (динамический dispatch). **Проверка:** малые батчи +
fast-gate + характеризационный каждый батч.

---

## 3. Рекомендованный порядок

1. **P1** (цикл, 1 правка) → коммит.
2. **P2** (self-imports, по модулю/коммит) → выпрямляет ещё рёбра.
3. Перепроверить граф скриптом из §1 — сколько циклов осталось.
4. **P3** (вынести хелперы) — самый большой выигрыш по «облегчить импорты».
5. **P5** (deep dead-code) — ещё ужать LOC перед разбиением.
6. **P4** (split neuron.py / preprocess_neuron) — в последнюю очередь, ради читаемости.

Бэкстоп на каждом шаге: fast-gate (~5 c) до, характеризационный тест (~11 мин) на батч.
Команда сборки графа — в истории сессии (скрипт на Python, regex `^from \. import`).
