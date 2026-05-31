# NEURD — slim-пайплайн сегментации: как работает + патчи

Что геометрически происходит на пути `mesh → Neuron`, где это в коде, что можно/нельзя
трогать, и какие compat-патчи держат всё на numpy2/trimesh4 без Docker.
Карта модулей — в [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md).

> ⚠️ Downstream-стадии (multi-soma split, cell typing, axon, autoproof, after-proof stats)
> и их модули **удалены** из форка. Здесь описан только живой slim-путь (2 стадии).

---

## 0. Картина целиком

Вход — **меш одного нейрона** из EM: огромная треугольная поверхность (сотни тысяч граней) —
округлое **тело (сома)** с ветвящимися **отростками** (дендриты/аксон), покрытыми **шипиками**.
Задача slim-пайплайна — превратить «мешок треугольников» в структуру: найти сому, разложить
отростки на **скелет** (центрлайны) и **ветки**, построить **граф связности (concept network)**,
снять сырые шипики. Три примитива, вокруг которых всё крутится: **SDF**, **скелет**, **concept network**.

Каноничный путь — [segmentation_pipeline.py](neurd/segmentation_pipeline.py):
1. **Идентификация сомы** — `sm.soma_indentification(mesh)`.
2. **Декомпозиция** — `neuron.Neuron(mesh=...)` (внутри `preprocess_neuron`: скелетонизация,
   ветви, сырые шипики, concept network).

`process_all_neurons.py` (entry пользователя) идёт ещё короче — `neuron.Neuron(mesh=...)`
напрямую в воркер-процессе на каждый OFF-меш. Stats-шаг `calculate_decomposition_products`
**вне** slim-пути (его единственного вызывателя убрали; частично починен).

---

## 1. Геометрические примитивы

- **SDF (Shape Diameter Function, «толщина»).** Для каждой грани луч **внутрь** меша по
  инвертированной нормали; длина до выхода = локальная толщина. Сома толстая → высокий SDF;
  дендриты/аксон тонкие → низкий. Главный признак «сома vs отростки» и оценки ширины веток.
  **Код:** `mesh_tools.trimesh_utils.ray_trace_distance(mesh)` (embree если есть). ⚠️ В пайплайне
  SDF ожидается **нормализованным в [0,1]** — пороги (`soma_width_threshold=0.32`) под эту шкалу.
- **Скелет (skeleton).** 1D-граф по «середине» трубчатого отростка (осевая линия). Длина =
  длина отростка; точки несут ширину (из SDF); разбивается на **ветки** в точках ветвления.
  **Код:** скелетонизация в `preprocess_neuron.py` (meshafterparty/MAP, `mesh_tools.skeleton_utils`).
- **Mesh segmentation по SDF.** Кластеризация граней по SDF. Оригинал — CGAL MRF graph-cut;
  наш stub (`neurd/_cgal_segmentation.py`) — KMeans по log(SDF) без пространственного сглаживания
  (контраст SDF сома/отростки велик → достаточно). **Код:** `tu.mesh_segmentation(mesh, clusters,
  smoothness)` → sub-меши + **SDF-медиана сегмента** (сома = высокая медиана + подходящий размер).
- **Concept network.** Направленный граф веток лимба: узлы=ветки, рёбра=«A продолжается в B»,
  корень = точка касания сомы, направление upstream→downstream (от сомы наружу). Один граф на
  (лимб, сома). **Код:** `nru.branches_to_concept_network(...)`, живёт в `Limb.concept_network`.

### Иерархия объекта Neuron
```
Neuron
 ├── somas:  S0, S1, ...                    (тела: меш + центр)
 └── limbs:  L0, L1, ...                    (отростки = связные компоненты после вырезания сомы)
       ├── branches: 0,1,2,...              (участки скелета между ветвлениями)
       │     ├── mesh, mesh_face_idx        (подмеш ветки)
       │     ├── skeleton                    (центрлайн)
       │     ├── width_array (из SDF)
       │     ├── endpoint_upstream/downstream
       │     └── web                         (меш-«перепонка» в точке ветвления)
       └── concept_network                   (граф веток, корень = касание сомы)
```
Файлы: `neuron.py` (классы), `neuron_utils.py` (`nru.*` — запросы по графу), `branch_utils.py`,
`limb_utils.py`, `concept_network_utils.py`.

---

## 2. Две живые стадии — геометрия и код

### Стадия 1 — Soma identification (`soma_extraction_utils`)
Самая геометрически плотная. Найти меш(и) сомы:
1. Decimation (грубее) → быстрый кандидат (`meshlab.Decimator`).
2. Выделить крупные куски меша.
3. **Poisson surface reconstruction** (`meshlab.Poisson`) → водонепроницаемая оболочка.
4. **Remove interior** (`meshlab.Interior`, через Ambient Occlusion).
5. **Mesh segmentation по SDF** (`tu.mesh_segmentation`, clusters≈3) → сегменты + SDF-медианы.
6. **Отбор:** сегмент проходит если `SDF_median > soma_width_threshold` (0.32) **и** размер в
   `[soma_size_threshold, _max]` → «толстый и крупный округлый кусок».
7. **Sphere validator** (bbox ≈ шар) + **backtrack** грубой оболочки на грани оригинального меша.

⚠️ Порог `0.32` и отбор завязаны на нормализованный [0,1] SDF — любой новый поставщик SDF
**обязан** нормализовать (stub делает, перцентили [2,98]). Узкое место замены — шаги 3–4 (MeshLab).

### Стадия 2 — Decomposition (`neuron.Neuron(...)` → `preprocess_neuron.preprocess_neuron`)
1. Soma detection (переиспользует стадию 1, если не передана).
2. **Вырезать сому** → остаток распадается на **лимбы** (связные компоненты).
3. **Скелетонизация** каждого лимба (meshafterparty/MAP) → центрлайны.
4. **Разбить скелет на ветки** в точках ветвления → `limb_correspondence` (branch_mesh,
   branch_skeleton, width_from_skeleton на ветку).
5. **Ширины** веток из SDF вдоль скелета. 6. **Concept network** на лимб.

⚠️ Сердце core clump. `Branch.__init__` делает deepcopy submesh/skeleton — дорого по RAM
(перф ниже), но менять рискованно. Скелетонизация (meshparty) — отдельная зависимость, НЕ meshlab.

### Инварианты — НЕ менять без сквозного теста
- Пороги отбора сомы (`soma_width_threshold=0.32`, size thresholds) — завязаны на [0,1] SDF.
- `Branch`/`Limb`/`Neuron` и concept network — на структуре графа/атрибутах держится всё.
- Семантика направления concept network (upstream/downstream от сомы).

---

## 3. Применённые compat-патчи (numpy2 / trimesh4 / мёртвый meshlabserver)

Все — в `neurd/` (не upstream). Большинство — monkeypatch'и в
[neurd/__init__.py](neurd/__init__.py), применяются при `import neurd` (до импорта mesh_tools).
**Это load-bearing — понимать назначение перед правкой `__init__.py`.**

| # | Симптом | Причина | Фикс (где) |
|---|---|---|---|
| 1 | `TypeError ... scalar index` (soma split) | trimesh≥4 `mesh.split()` → `list`, не `ndarray` | `soma_extraction_utils.py`: `list(...)` + list-comprehension |
| 2 | Poisson не выполняется, нет выходного файла | `meshlab.Poisson` пишет `<xmlfilter>` XML, MeshLabServer 2020.09 игнорирует | `__init__.py`: патч `Poisson.initialize_script_filters` → `type=Rich*` |
| 3 | `NameError: csm` (CGAL не установлен) | C++ расширение CGAL отсутствует | `__init__.py`: stub `_cgal_segmentation.py` в `sys.modules['cgal_Segmentation_Module']` |
| 4 | `scipy ValueError: axis 0 index ... exceeds` | MeshLabServer 2020.09 OFF-экспортёр: компактные вершины, грани в старой нумерации | `__init__.py`: патч `Meshlab.fetch_mesh_from_off` (перенумерация searchsorted) |
| 5 | `IndexError ... size 1` (multi-soma split) | две сомы на одном стартовом узле → путь из 1 узла | `proofreading_utils.py:1147` guard *(модуль удалён; патч исторический)* |
| 6 | `ValueError: data_pts ... 2 dimensions` | новая pykdtree требует 2D, upstream строит из 1D | `__init__.py`: обёртка `skeleton_utils.KDTree` (1D→(N,1)) |
| 7 | `numpy_dep has no attribute 'in1d'` | `np.in1d` удалён в numpy 2 | `__init__.py`: `numpy.in1d = isin` + `numpy_dep.in1d = isin` |

> Корни багов — в `git log` соответствующих правок. Здесь — что/где, чтобы понимать `__init__.py`.

---

## 4. Forward: оптимизация — см. [OPTIMIZATION.md](OPTIMIZATION.md)

Полный workstream оптимизации (профиль, отгруженные выигрыши, план, GPU) вынесен в
[OPTIMIZATION.md](OPTIMIZATION.md). Кратко — что **замерено** (профиль 2026-05-30, не гипотезы):

- **MeshLab сабпроцесс ~74% времени.** Из них **Poisson был сломанным no-op** (выход = вход) и
  стоил ~536s — уже заменён in-process pass-through: пайплайн **688s → 151s (4.5×)**.
- **SDF-лучи 0.46s, deepcopy ~2s** — НЕ драйверы времени (прежние «гипотезы» про них неверны;
  deepcopy может быть драйвером RAM, но не скорости).
- Остаток meshlab для замены in-process: **FillHoles** (тоже сломан → no-op), **Decimator**
  (→ open3d, fidelity проверена), **Interior removal** (реальный, сложнее). Делать по одной,
  валидируя per-op фикстурой + характеризационным тестом.
- Реальный compute-пол: **скелетонизация** (meshparty/CGAL teasar) + **детект шипиков**.
- RAM (наблюдалось ~32 ГБ): deepcopy submesh/Branch + deepcopy Neuron на границах — для памяти и
  throughput (батч многих нейронов), не для latency одного.
</content>
