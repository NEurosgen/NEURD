# NEURD — оптимизация сегментации: профиль, выигрыши, план

Ветка `optimize_segmentation`. Цель: сократить время и RAM пути `mesh → Neuron`,
переписывая техники там, где это оправдано **замером** (а не интуицией). Метод —
measure-first: профиль → захват эталона → секундная проверка → 2.5-мин гейт.

**Целевой кейс:** одно-нейронные меши с **ровно одной сомой** (microns и h01). Оптимизации
могут это предполагать. ⚠️ Committed fixture `864691...` — это **два** нейрона (нерепрезентативен);
одно-сомный меш для валидации: `Applications/.../neuron_2530864375.off` (H01 → `data_type="h01"`,
иначе скелетонизатор падает `NameError: calcification_param`). Полный пайплайн на нём 215.8s
(сома 70s + декомпозиция ~146s), 1 сома.

Как работает пайплайн геометрически — [PIPELINE.md](PIPELINE.md). Карта модулей —
[NEURD_STRUCTURE.md](NEURD_STRUCTURE.md).

---

## 1. Baseline и результаты профайлера (2026-05-30, fixture-меш 323k граней)

**Чистый wall-clock (без профайлера):** сома 283.5s + декомпозиция 398.5s = **682s**, limbs=7.

**cProfile — относительная разбивка** (абсолюты раздуты overhead'ом профайлера + параллельной
нагрузкой; важны **пропорции**):

| Бакет | Доля / время | Природа |
|---|---|---|
| **MeshLab сабпроцесс** | **~74%** | спавн `xvfb`+`meshlabserver` + ASCII-OFF на диск, на крошечных мешах |
| └ **Poisson** | **~536s** (9 × ~60s) | **сломанный no-op** — выход байт-в-байт = вход (Screened Poisson не работает на MeshLabServer 2020.09) |
| Скелетонизация (meshparty/CGAL teasar) | ~178s | реальный граф-алгоритм |
| Детект шипиков (`calculate_spines`) | ~98s | `mesh_segmentation` (SDF) + trimesh stitch/holes/normals (Python) |
| SDF-лучи (`ray_trace_distance`) | **0.46s** | НЕ бочтлнек |
| deepcopy | **~2s** | НЕзначим по времени (возможно по RAM) |

**Что НЕ сработало (measure-first отсёк бесполезное):**
1. **GPU-SDF снят** — лучи уже 0.46s, ускорять нечего.
2. **deepcopy по времени — пшик (2s)** — не драйвер времени (может быть драйвером RAM).
3. **single-soma short-circuit (`max_somas`) — НЕ ускоряет** одно-нейронный меш: там ОДИН кусок,
   пропускать нечего (70.3s vs 70.1s). Multi-piece overhead есть только у многонейронных мешей.
   Параметр оставлен как **корректностный guardrail** (защита от over-segmentation), не как скорость.
4. **Бочтлнек — оверхед сабпроцесса**, и крупнейшее — Poisson, который вообще ничего не делал.

---

## 2. Отгруженные выигрыши

| Коммит | Что | Результат |
|---|---|---|
| `469f492` | **Poisson → in-process no-op** (выход = вход, без сабпроцесса) | пайплайн **688s → 151s (4.5×)**; сома **283s → 45s**; контракт 4 passed |

Poisson был байт-идентичным no-op (доказано захватом пар вход/выход), но платил ~536s за
спавн. Соме-детект работает без него (sphere validator терпит non-watertight). Замена
behavior-preserving — характеризационный тест подтвердил идентичность контракта.

---

## 3. Метод (быстрый цикл вместо 11 минут)

- **Захват эталона:** [tests/tools/capture_mesh_op_fixtures.py](tests/tools/capture_mesh_op_fixtures.py)
  monkeypatch'ит meshlab-операции и пишет реальные пары вход/выход (фикстуры gitignored, регенерируемы).
- **In-process операции:** [neurd/_mesh_ops.py](neurd/_mesh_ops.py) — decimate (open3d quadric),
  poisson (open3d screened, для опц. эксперимента качества), fill_holes (trimesh).
- **Секундные проверки:** [tests/unit/test_mesh_ops.py](tests/unit/test_mesh_ops.py) (синтетика, ~3s)
  + [tests/integration/test_mesh_ops_fidelity.py](tests/integration/test_mesh_ops_fidelity.py)
  (vs meshlab-эталон, ~13s; decimate validated).
- **Гейт:** характеризационный тест `tests/integration/test_segmentation_pipeline.py` (теперь ~2.5 мин).

---

## 4. План вперёд (по убыванию замеренной отдачи)

Осталось **151s = сома ~45s + декомпозиция ~106s**.

| # | Цель | Где | Сложность | Ожидание |
|---|---|---|---|---|
| 1 | **fill_holes → in-process no-op** | `__init__.py` патч | тривиально/безопасно | тоже сломан (returncode 255 → «continuing without»); убрать ~3 спавна |
| 2 | **Decimator → open3d** | `soma_extraction_utils` / патч | низко (fidelity ✓) | убрать спавны децимации |
| 3 | **remove_interior** (meshlab Interior) | разобраться | средне | реальная операция, разобрать backend |
| 4 | **Скелетонизация** (meshparty/CGAL teasar) | `preprocess_neuron` | высоко | это **пол** latency одного нейрона; граф-алгоритм |
| 5 | **Детект шипиков** | `spine_utils` + trimesh ops | средне | stitch/holes/normals — Python-тяжёлые, частично векторизуемо |
| 6 | **RAM** (deepcopy submesh/Branch; deepcopy Neuron на границах) | `neuron.py` | средне-высоко | для **памяти/throughput**, не для latency одного нейрона |

После #1–#3 (остатки meshlab) пол времени = реальный compute: скелетонизация (#4) + шипики (#5).

---

## 5. GPU — честная оценка (idea пользователя, держим в уме)

GPU помогает там, где **плотная параллельная численная работа**; не помогает там, где оверхед
сабпроцесса (это лечится in-process) или граф-алгоритмы.

- **Single-neuron latency — ограниченный потенциал GPU.** Когда meshlab уйдёт, пол времени —
  скелетонизация (teasar = обход графа, плохо ложится на GPU) и mesh-topology операции. SDF уже
  0.46s. То есть для **одного нейрона** GPU мало что даст после in-process замен.
- **Throughput — вот где GPU реально играет.** `process_all_neurons` гоняет **много** мешей.
  Если сократить RAM на нейрон (#6), можно батчить больше нейронов параллельно, и GPU ускорит
  параллельно-дружелюбные куски (SDF/`ray_trace`, KMeans-сегментация) **через батч**. Поэтому
  **GPU в паре с оптимизацией памяти даёт пропускную способность**, а не скорость одного меша.
- **Что GPU-able по частям:** ray-casting/SDF (open3d RaycastingScene / warp / OptiX),
  KMeans SDF-сегментации (cuML). Скелетонизация и mesh-stitching — нет.
- **Предусловие:** CUDA-Python стек (torch/warp/cupy) не установлен; GPU есть (RTX 5060 = sm_120,
  новая → совместимость колёс проверить). Браться за GPU **после** in-process замен и RAM (#1–#6),
  когда станет ясно, latency или throughput мы упёрлись.

**Итог направления:** доделать in-process замены meshlab (#1–#3, дешёво и безопасно) → затем
RAM (#6) → затем решать GPU как throughput-рычаг для пакетной обработки. Скелетонизация (#4) —
отдельный тяжёлый фронт, если single-neuron latency останется критичной.
