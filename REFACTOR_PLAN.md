# NEURD — план чистки: сделано + что осталось (актуально 2026-05-30)

Состояние: `neurd/` = **16 файлов, ~24.4k LOC**. Slim-пайплайн `mesh → Neuron` зелёный
(unit 59 passed / 1 skipped; характеризационный тест 4 passed, ~11 мин). Граф внутренних
импортов — **DAG (0 циклов)**. Глубокий dead-code — см. [DEAD_CODE_REACHABILITY.md](DEAD_CODE_REACHABILITY.md).

---

## 1. Как перепроверять граф импортов

Скрипт-обход (regex по заголовкам файлов), ребро `A→B` если `A` импортирует `B`:
`^from \. import (\w+)`, `^from \.(\w+) import`, `^from neurd import (\w+)`,
`^from neurd\.(\w+) import`. Затем DFS на циклы / 2-cycle детектор. До рефактора было
6 двусторонних циклов + god-hub; сейчас DAG. Если правки снова заведут цикл — этот обход покажет.

---

## 2. Сделано (2026-05-30)

- **P1 — разорван корневой цикл `neuron_utils ↔ neuron`.** `neuron_utils` (god-hub, импортируется
  9 модулями) тянул `neuron.Branch` в одном месте → локальный импорт. Коммит `refactor(P1)`.
- **P2 — убраны все 7 self-import'ов** (`from . import <self> as alias`). 5 модулей → прямые
  вызовы; `concept_network_utils`/`spine_utils` → `alias = sys.modules[__name__]` (в call-site'ах
  есть локальные имена-тени, слепой strip сломал бы); `limb_utils`/`soma_extraction_utils` —
  где модуль-объект передаётся аргументом, явная self-ссылка. Коммит `refactor(P2)`.
- **P3 — `neuron_utils` сделан intra-package sink'ом + устранены ВСЕ циклы (6→0, DAG).**
  Лазифицированы 3 ссылки neuron_utils (cnu/nst) + одноразовые `neuron→preprocess_neuron`,
  `cnu→nst`; сняты 3 мёртвых перекрёстных импорта (0 использований). Коммиты `refactor(P3)`.
- **P5 — свежий zero-ref sweep (−424 LOC, 21 функция, 3 волны).** Манульная вычистка пользователя
  (удаление `stats_df` и др.) осиротила батч; срезано до фикспоинта. Коммит `dead-code(P5)`.
- **P4 — ОТМЕНЁН** пользователем: split больших файлов на пакеты улучшает только читаемость,
  импорты не упрощает (классы `neuron.py` ссылаются друг на друга и на `nru`). В план не входит.

---

## 3. Что осталось (по убыванию ценности)

1. **Глубокий reachability dead-code** (mark-and-sweep от корней) — zero-ref исчерпан, но
   взаимно-ссылающиеся мёртвые кластеры им не ловятся. Метод, корни, динамические хазарды,
   подводные камни — в [DEAD_CODE_REACHABILITY.md](DEAD_CODE_REACHABILITY.md). Самый вероятный
   улов: `calculate_decomposition_products` + его stats-цепочка (вне slim-пути). ⚠️ Рискованно
   из-за динамического dispatch — нужен характеризационный тест **на каждый батч**.
2. **Поднять bottom-of-file импорты наверх.** Циклов больше нет → band-aid-импорты в конце
   файлов (`#--- from neurd_packages ---`) можно перенести в шапку. Косметика, но снимает
   историческую хрупкость. Отдельными коммитами, fast-gate каждый.
3. **Оптимизация скорости/RAM** — отдельный workstream на ветке `optimize_segmentation`,
   расписан в [OPTIMIZATION.md](OPTIMIZATION.md). Уже: Poisson-no-op дал пайплайн 688s → 151s
   (4.5×). Дальше: FillHoles/Decimator/Interior in-process, скелетонизация, шипики, RAM, GPU.

Бэкстоп на каждом шаге: fast-gate (~4 c) + характеризационный тест (теперь ~2.5 мин) на батч.
</content>
