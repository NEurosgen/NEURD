# NEURD — план рефактора читаемости

Цель: убрать мусор, не меняя поведение. Каждый файл — отдельный коммит.
Backstop: `python -m pytest tests/unit/` (fast-gate, ~4s) после каждого файла.

Предыдущий структурный рефактор (import-циклы, dead-code) — в `git log`; результаты
зафиксированы в `main`. Здесь — только читаемость.

> **Статус 2026-07-22.** Малые/средние файлы §2 в основном сделаны: `width_utils` (457→183),
> `branch_utils` (782→355, + skeletal_coords декаплинг), `concept_network_utils` (`554dc95`),
> `neuron_statistics` (`42798c2`); `limb_utils` больше нет отдельным файлом. Крупные файлы (§2⑤) —
> частично: `neuron.py` классы декомпозированы (3403→1817, memory `neuron-classes-cleanup`),
> `preprocess_neuron` фазирован. Остаются: `spine_utils` (COLD, нужен spine-harness — SPINE_UTILS_PLAN),
> хвосты `neuron_utils`/`preprocess_neuron`. Типовые смеллы §1 и AST-метод §2⑤ — durable, применять дальше.

---

## 1. Типовые смеллы (встречаются во всех файлах)

| Смелл | Пример | Что делать |
|---|---|---|
| **Bottom imports** | `#--- from neurd_packages ---` в конце файла | Поднять в шапку |
| **Дублирующиеся импорты** | `numpy_dep as np` на строке 3 и 107 | Оставить один |
| **Мёртвые импорты** | `import time` / `ipyvolume_utils` без использования | Удалить |
| **`verbose` / `print_flag` спам** | `if verbose: print(f"...")` в 5–10 местах функции | Удалить print'ы; `verbose` убрать из сигнатуры |
| **Закомментированный код** | `# if print_flag:` / `# print(f"...")` | Удалить |
| **No-op присваивание** | `distance_threshold = distance_threshold` | Удалить |
| **Многострочные «example» docstring** | 15+ строк с примером вызова | Заменить одной строкой «что делает» |
| **Пустые блоки** | 10+ подряд пустых строк | Сократить до одной |

---

## 2. Файлы в порядке приоритета

### ①②③ `limb_utils` / `width_utils` (457→183) / `branch_utils` (782→355) ✅ DONE

Малые/средние файлы вычищены (мёртвые/дублирующиеся импорты, bottom-imports в шапку,
`verbose`/`print_flag` спам, длинные example-docstring). `limb_utils` больше не отдельным файлом;
`branch_utils` дополнительно получил skeletal_coords декаплинг. Типовой рецепт — таблица смеллов §1.

---

### ④ `concept_network_utils.py` — 1325 → 849 LOC ✅ DONE (commit 554dc95)

Граф-утилиты. Сделано: импорты в шапку (мёртвый `wu`, дубль `np`, `numpy_dep`→`numpy`);
снят весь `verbose`/`if verbose: print` спам и `verbose=verbose` пробросы (verbose нигде
не в логике, внешних keyword-вызывателей нет); удалены 4 мёртвых `'''`-блока
закомментированных функций (~227 строк). Поведение не изменено.

---

### ④·5 `neuron_statistics.py` — 1951 → 1679 LOC ✅ DONE (commit 42798c2)

Импорты в шапку (dedup gu/np, numpy_dep→numpy; `nst` остаётся локальным cycle-breaker).
Убраны 33 чистых print-`if verbose:` блока + пробросы; `verbose`-параметр снят у 19 функций,
сохранён у 7 (реально используют). Метод — **AST-трансформер** (см. ниже), безопаснее регекспов.

### ⑤ Крупные файлы (отдельная сессия)

`neuron_utils.py` (3462), `neuron.py` (3423), `preprocess_neuron.py` (3175 — `preprocess_neuron`/
`preprocess_limb` уже декомпозированы на именованные фазы, см. PIPELINE.md §2), `spine_utils.py`
(3323) — только после того, как выработан ритм на малых. Риск выше: критический путь пайплайна.

**AST-метод для verbose-чистки (выработан на concept_network/neuron_statistics):**
1. `tokenize` → множество строк внутри multi-line строк (docstring/`'''`-блоки) — не трогать.
2. `ast`: удалять `if verbose:`-блоки, только если тело «чистый print» (рекурсивно: лишь
   Expr-print/Constant/Pass и циклы из них; никаких Assign/Return/Raise/With/Try).
3. `verbose`-параметр снимать у функции, ТОЛЬКО если в её теле не осталось ни одного
   `verbose` (Load) после удаления блоков и пробросов. Иначе оставить.
4. Финальная AST-проверка: каждый оставшийся `verbose`(Load) имеет параметр в своей функции
   (ловит NameError до тестов). Затем unit + integration.
Backstop-комбо: `py_compile` + AST scope-check + `pytest tests/unit` + integration (~95s).

---

## 3. Конвенции

- **Не менять поведение.** Rename-only и удаление мёртвого кода — ok. Логика — нет.
- **Один файл = один коммит.** `refactor(читаемость): <имя_файла> — что убрано`.
- **Fast-gate после каждого коммита** (`pytest tests/unit/`).
- **`verbose` / `print_flag` убирать целиком**, включая параметр из сигнатуры — вызывающие
  передают его через `**kwargs` и не заметят удаления.
- **Bottom imports → шапка:** перенести блок `#--- from neurd_packages ---` в начало,
  убрав комментарий-разделитель.
- **Не трогать `__init__.py`** — compat-патчи там хрупкие, у каждого patch-блока своя причина.
