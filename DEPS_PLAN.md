# NEURD — План расцепления и удаления downstream-кластера (Фаза 5)

Цель проекта: **оставить только то, что нужно для сегментации меша** — на выходе
`Neuron` с сомами + ветками (mesh + skeleton) + шипиками. «Типы клетки» и «графы»
(autoproof, cell typing, синапсы, статистика) не нужны.

Слим-вход уже есть: [neurd/segmentation_pipeline.py](neurd/segmentation_pipeline.py)
гоняет только нужные стадии. Геометрия и инварианты — в
[PIPELINE_GEOMETRY.md](PIPELINE_GEOMETRY.md), карта стадий — в [PIPELINE.md](PIPELINE.md).

---

## Сделано ранее (кратко)
- **Фазы 1–4:** Docker→локальный Python 3.12 / numpy2 / trimesh4; удалены GNN,
  connectome, motif, proximity (~16k LOC); `extras_require`; numpy-шим.
- **Пайплайн зелёный (2026-05-28):** 8 compat-багфиксов (CGAL→питон-stub, meshlab OFF,
  multi-soma guard, pykdtree-1D, in1d, dotmotif). Все патчи — в `neurd/`. Полный
  автопруф-пайплайн проходил 9/9 стадий; добавлен slim-оркестратор.

---

## Фаза 5: relocate-then-delete downstream-кластера

### Стратегия (почему именно так)
Помодульное удаление «в лоб» **проваливается** (проверено): модули кластера импортят
друг друга, и часть имеет **живые функции**, которые зовёт ядро. Поэтому:

1. **Прорежение ядра первым.** Бóльшая часть ссылок ядра на кластер — внутри **мёртвых
   методов** (синапсы/autoproof/типы/статистика), которые на слим-пути не исполняются.
   Удаляем эти мёртвые методы из оставляемых модулей → ссылки на кластер исчезают.
2. **Релокация немногих живых функций.** То, что реально исполняется на слим
   (coverage-live), переносим в оставляемые модули или в новый маленький модуль.
3. **Удаление кластера** целиком + снятие импортов. Gate после каждой волны.

### Два числа на модуль (из coverage + статики)
- **live** = функций реально исполнилось на slim-прогоне (`/tmp/cov.json`, fixture-меш).
  Это то, что **точно надо сохранить** (релоцировать).
- **kept-refs** = функций кластера, на которые **статически ссылается остающийся код**.
  Бóльшая часть — в мёртвых методах ядра (исчезнут при прорежении), но каждую надо
  подтвердить (живая ссылка → релоцировать; мёртвая → удалить вместе с методом).

> ⚠️ Coverage снят на **одном** fixture-меше. «0 live» для семантически-downstream
> модулей (autoproof/синапсы/типы/стат) надёжно. Но перед удалением каждой функции —
> grep на вызовы; при сомнении сохранять. В идеале добрать покрытие на 2–3 разных мешах.

### Кластер на удаление (19 модулей, ~40k LOC) и порядок волн

| Волна | Модули | live | kept-refs | Заметки |
|---|---|---|---|---|
| **5.1 graph/autoproof** | graph_error_detector(+_axon/_dendrite), graph_filter_pipeline, neuron_graph_lite_utils, microns_graph_query_utils, neuron_geometry_utils | 0/0/0/0/0 | 0 | ссылаются только внутри кластера — падают вместе с 5.2 |
| | graph_filters | 0 | 7 (`gf.*_filter`) | в мёртвых autoproof-методах |
| | error_detection | 0 | 2 (`ed.matched_branches_by_angle`, `width_jump_from_upstream_min`) | мёртвые методы |
| | proofreading_utils | (import-time) | 6 (`pru.cut_limb_network_by_suggestions`, `merge_*`, `v7_*_filters`) | autoproof + multi-soma (мёртвые на слим) |
| | soma_splitting_utils | 0 | 3 (`ssu.calculate_multi_soma_split_suggestions`/`multi_soma_split_execution`/…) | multi-soma split (не нужен — один нейрон) |
| | neuron_pipeline_utils | 0 | 0 | старый оркестратор; ссылок в ядре нет (только тест) |
| **5.2 synapse_utils** | synapse_utils | **0** | **51** | 0 живых! Все 51 — в мёртвых синапс-методах ядра. Прорезать методы → удалить |
| **5.3 typing** | axon_utils | 1 (`dendrite_limb_branch_dict`) | 7 | релоцировать live label-хелпер; остальное мёртвое |
| | classification_utils | 1 (`axon_limb_branch_dict`) | 10 | то же |
| | apical_utils | 1 (`dendrite_compartment_labels`) | 9 | то же |
| | cell_type_utils | 3 (model loaders) | 3 | E/I-классификация; на слим не нужна — проверить, мёртвые ли live |
| **5.4 neuron_statistics** | neuron_statistics | **10** | 45 | **реально нужен** (зовётся `neuron_searching`/spine-путём — на нём упал Batch 1). Релоцировать 10 live в `neuron_stats_lite` или `neuron_utils` |
| **5.5 branch_attr_utils** | branch_attr_utils | 3 | 6 | релоцировать 3 live |

live-функции neuron_statistics (релокация-цель 5.4): `skeletal_length_along_path`,
`total_upstream_skeletal_length`, `distance_from_soma`, `stats_dict_over_limb_branch`,
`features_from_neuron_skeleton_and_soma_center`, `features_from_skeleton_and_soma_center`,
`centroid_stats_from_neuron_obj`, `skeleton_stats_from_neuron_obj`,
`limb_branch_from_stats_df`, `neuron_stats`. (Транзитивные зависимости внутри nst — проверить.)

### Затрагиваемые оставляемые модули (где прорезать мёртвые методы)
Ссылки на кластер живут в: `neuron.py` (syu ×50, apu, nst, au, clu, ssu, pru),
`neuron_utils.py` (nst, syu, clu, apu, au, ed), `spine_utils.py` (apu, syu, ctu, nst),
`branch_utils.py` (syu, au, nst, bau), `concept_network_utils.py` (nst, au),
`limb_utils.py` (au, nst), `neuron_searching.py` (au, clu, ed, nst, syu — **часть живая!**),
`neuron_visualizations.py` (au, pru, syu — под `plot_*` флагами), `vdi_default.py`/`vdi_h01.py`
(gf, pru, syu), `volume_utils.py`/`h01_volume_utils.py` (syu).

### Процедура на каждую волну (инкрементально, с откатом)
1. Снять с coverage список live-функций затрагиваемых модулей (есть в `/tmp/cov.json`;
   при необходимости перегенерировать).
2. **Релоцировать** live-функции кластера данной волны в оставляемый модуль / новый
   `neuron_stats_lite.py` (для 5.4). Переписать вызовы в ядре на новый источник.
3. **Удалить мёртвые методы** ядра, ссылающиеся на кластер этой волны (по coverage-dead
   списку; каждую — grep на живых вызывающих перед удалением).
4. Снять импорты кластера + удалить файлы кластера данной волны.
5. **Gate:** `import neurd` + `pytest tests/unit/` + slim-прогон
   (`tests/integration/test_segmentation_pipeline.py`) — всё зелёное.
6. Коммит на форк (`git push origin refactor_dependens`) как чекпойнт волны.

### Якорь (safety net)
- Юнит-тесты (`tests/unit/`, 84 passed) + оракул CGAL
  (`tests/integration/test_cgal_segmentation_oracle.py`).
- **Slim-характеризационный тест** `tests/integration/test_segmentation_pipeline.py`
  (был создан в попытке Batch 1, нужно восстановить): на fixture-меше проверяет
  somas≥1, limbs>0, у каждой ветки есть mesh+skeleton. Это контракт выхода —
  обязателен ПЕРЕД прорежением ядра.

### Урок неудачной попытки Batch 1 (2026-05-28)
Удалил 19 модулей «в лоб» без релокации/прорежения → `import` починился, но slim упал
на `NameError: nst` (neuron_searching звал neuron_statistics на spine-пути). Откатил
через `git checkout HEAD`. Вывод: **сначала прорежение ядра + релокация живого, потом
удаление**; и `neuron_statistics` НЕ пустой (10 live) — его нельзя удалять целиком.

---

## Чего НЕ трогаем
- Upstream-пакеты (`datasci_tools`, `mesh_tools`, `meshparty`) — только обёртки/шимы в `neurd/`.
- `meshparty`/CGAL-скелетонизация, `open3d` — ядро mesh-геометрии.
- Ядро декомпозиции: `neuron.py`/`preprocess_neuron`/`soma_extraction_utils`/
  `neuron_simplification`/`branch_utils`/`spine_utils` (прореживаем мёртвые методы,
  но НЕ ломаем декомпозицию/spine-геометрию).

## Метрики-цель Фазы 5
- LOC: ~84k → ~45k (удаляемо >половины: кластер ~40k + мёртвые методы ядра).
- Зависимости: уйдёт **dotmotif/grandiso/lark-parser** (с волной 5.1).
- Файлов `neurd/*.py`: 44 → ~25.
