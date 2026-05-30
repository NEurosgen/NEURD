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
| Скелетонизация (meshparty/CGAL teasar) | ~178s ⚠️ | **загрязнено** Poisson внутри (см. чистый профиль §4: реально ~69s, teasar 0.31s) |
| Детект шипиков (`calculate_spines`) | ~98s ⚠️ | на самом деле ДОМИНАНТА декомпозиции (см. §4: ~133s) |
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

## 2. Отгруженные выигрыши (коммиты на `optimize_segmentation`)

| Коммит | Что | Результат |
|---|---|---|
| `469f492` | **Poisson → in-process no-op** (meshlab Poisson был сломан, выход=вход) | пайплайн (microns 2-сомы) **688s → 151s (4.5×)**; сома 283s → 45s; контракт 4 passed |
| `5aca8c0` | **FillHoles → in-process no-op** (тоже сломан, returncode 255) | сома 45s → 41s; та же сома |
| `4982d0c` | **Neuron-output baseline** (safety net для output-changing замен) | `tests/fixtures/neuron_baseline.json` + `TestNeuronBaseline` |
| (committed) | **`max_somas` guardrail** (single-neuron) | корректность (не скорость — см. §1) |
| `fedde35` | **`copy_concept_network` — убран двойной deepcopy** | пиковые Branch-аллокации при копировании 2N → N |
| `2a86407` | **`preprocessed_data` очистка** — удалить `limb_meshes`/`limb_concept_networks`/`soma_meshes` после init | −190 МБ живых объектов; ключи были дубликатом данных уже в Limb/Soma |
| `3643d3e` | **trimesh-кэши** — очистить Limb.mesh + neuron.mesh после init | −380 МБ живых объектов (−83% от измеренных атрибутов) |
| `bdfd8b7` | **KMeans `n_init=10→1`** в `_cgal_segmentation.py` (наш Python-стенд CGAL) | fixture build **142s → 124.9s**, стадия шипиков 52.9s → 43.0s; **выход byte-identical** (1-D log(SDF) k-means++ сходится в тот же оптимум); baseline PASS. Бьёт по сому+шипикам (обе зовут `cgal_segmentation`) |
| (uncommitted) | **Ленивая шафт-рестрикция** — `restrict_meshes_to_shaft_meshes_without_coordinates` считает `mesh_volume`/`close_hole_area` (починка дыр) лениво в порядке cheap→expensive вместо жадного `stats_df` | стадия шипиков **43.0s → 14.3s (~3×)**, build 124.9s → **95.9s**; **byte-identical**: 28/28 вызовов `restrict` дали тот же выбор шафта, что жадный (вход не мутируется — `mesh_volume` читает, не меняет геометрию); baseline PASS |

Poisson/FillHoles были байт-идентичными no-op (доказано захватом пар вход/выход), но платили
~536s+ за спавн. Соме-детект работает без них. Замены behavior-preserving — характеризационный
тест подтвердил идентичность контракта. **Это главный результат по времени.**

RAM-сессия (2026-05-30): нейрон-объект похудел с **~458 МБ → ~78 МБ** live-size (замер через
`deep_size` на fixture 323k граней, 2-сомы, 7 лимбов, 49 веток). RSS-пик процесса (1760 МБ)
не изменился — это watermark; реальный выигрыш виден при форке воркеров.

⚠️ Тайминги выше — на microns 2-сомном fixture. На **целевом H01 одно-сомном** меше (507k граней)
полный пайплайн с этими фиксами = **215.8s** (сома 70s + декомпозиция 146s).

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

## 4. Чистый профиль декомпозиции (H01, Poisson-no-op активен) — где реально 146s

Перепрофилировано на целевом H01 меше (старый профиль был **загрязнён**: Poisson вызывался
ВНУТРИ скелетонизации, ~60s/вызов — теперь no-op):

| Высокоуровневая функция | Время | Природа |
|---|---|---|
| **`calculate_spines_on_neuron`** (детект шипиков) | **~133s (65%)** | per-branch (×28); НЕ логика (SDF-сегментация ~22s), а **починка меша**: объёмы (`fill_mesh_holes_with_fan`+`fix_normals`, ~5×/ветку) + `group_rows` (1М вызовов) |
| Скелетонизация (`preprocess_neuron`) | ~69s | ⚠️ **teasar-ядро = 0.31s** — смена алгоритма (Kimimaro/Skeletor) НЕ поможет; дорога обвязка (стичинг/графы) |

**Вывод:** узкое место декомпозиции — **шипики (133s)**, не скелетонизация. Логику шипиков
оптимизировать почти нечего (дёшево); дорога mesh-починка/объёмы (upstream `mesh_tools`,
output-рискованно). Пользователю шипики **нужны** (не отключить).

### 4a. Свежий per-function профиль стадии шипиков (2026-05-31, microns fixture, cProfile)

Разбивка `calculate_spines_on_branch` (×28 веток, ~53s чистыми; spines=0 на этом меше → чистая
**детекция** без объёмов реальных шипиков):

| Поддерево | cumtime | Что |
|---|---|---|
| `restrict_meshes_to_shaft_meshes_without_coordinates` → `query_meshes_from_stats` → `stats_df`/`stitch` → `mesh_volume` → `fill_mesh_holes_with_fan` → `fix_normals`/`fix_winding` | **~66s (65%)** | «шафт vs спайн» считает **объём+площадь дыр каждого сегмента** через починку. `fix_winding` 42.7s, `group_rows` 27.8s (427K вызовов) — внутренности trimesh repair |
| `mesh_segmentation` → `cgal_segmentation` → **sklearn KMeans** | 28.9s (KMeans 22s) | ✅ **ВЗЯТО:** `n_init=10→1`, см. §2 |

**✅ Зацепка #2 ВЗЯТА (output-preserving):** `restrict_meshes_to_shaft_meshes_without_coordinates`
теперь считает статистики лениво (cheap→expensive) вместо жадного `tu.stats_df`. Запрос
`(close_hole_area > X OR mesh_volume > Y) AND (n_faces > min)`: `close_hole_area` нужен только при
`n_faces>min`, `mesh_volume` — только при `n_faces>min И close_hole_area<=X`; остальным — 0-sentinel
(их решают другие термы, значение не влияет). Тот же query-evaluator → byte-identical (28/28
вызовов совпали с жадным; вход не мутируется — проверено: `mesh_volume`/`stitch` пишут в **новый**
меш). **43s → 14.3s (~3×)**, см. §2.

**Следующая зацепка (#3, не разобрана):** что осталось в 14.3s стадии шипиков — пере-профилировать
(вероятно `mesh_segmentation`/CGAL-диск-roundtrip + `close_hole_area` на выживших). Также width
(§отдельно): считается ~дважды по всем веткам (`median_mesh_center` внутри шипиков +
`no_spine_median_mesh_center` после), `branch_mesh_no_spines` пересоздаётся на вызов.

---

## 5. Plan вперёд (по убыванию замеренной отдачи) + СТЕНА

| # | Цель | Где | Сложность | Статус/ожидание |
|---|---|---|---|---|
| 1 | ~~fill_holes → no-op~~ | `__init__.py` | — | ✅ **СДЕЛАНО** (`5aca8c0`) |
| 2 | **Decimator → open3d** | патч `__init__.py` | низко (fidelity ✓) | **держим** — output-changing: меняет сому (median 0.533→0.479), −12s. Прототип был, откатан. Судить против baseline |
| 3 | **remove_interior** (meshlab `Interior`, ambient occlusion) | соме-стадия | высоко | реальная операция (не no-op!), прямого in-process аналога нет — писать через рейкастинг (open3d RaycastingScene) |
| 4 | **Шипики 133s — параллелизация по веткам** | `spine_utils.calculate_spines_on_neuron` | **высоко** | ⛔ **ОТЛОЖЕНО (2026-05-31):** `fork` даёт **deadlock** (унаследованные потоки BLAS/CGAL), а `forkserver`/`spawn` раздувают RAM ~3×. Блокер `br.spines` снят, контракт ясен — но безопасной дешёвой по RAM реализации нет. См. ниже |
| 5 | ~~**RAM** (deepcopy/кэши)~~ | `neuron.py` | — | ✅ **СДЕЛАНО** (`fedde35`, `2a86407`, `3643d3e`) — live-size 458 МБ → 78 МБ |

### 🧱 СТЕНА: параллелизация шипиков (попытка провалена, откатана; частично снята)

Форк-`multiprocessing.Pool` **взрывал память**: нейрон ~1.7 ГБ RSS, ~18 форк-процессов × 1.7 ГБ
→ своп → медленнее последовательного.

**Текущее состояние после RAM-оптимизаций:**
- Live-size нейрон-объекта: **~78 МБ** (было ~458 МБ)
- RSS процесса всё ещё ~1.7 ГБ (watermark; Python + библиотеки + историческая аллокация)
- При форке воркеры наследуют **текущие** страницы, не watermark → COW-давление снижено

**Два пути к параллелизации:**
- **Путь A (рекомендуется):** теперь нейрон лёгкий — попробовать форк снова, замерить RSS per-worker.
  Ожидание: воркеры наследуют ~78 МБ Python-объектов вместо 458 МБ → мало COW.
- **Путь B (запасной):** пиклить воркерам только **меши веток** как numpy arrays, не весь нейрон.

✅ **БЛОКЕР СНЯТ (2026-05-31): где хранятся спайны и почему читались 0.**

Спайны живут **на `Branch`** как обычные изменяемые атрибуты (НЕ property):
- `branch.spines` — list submesh-объектов, выход `calculate_spines_on_branch`, пишется в
  `spine_utils.py:2214`.
- `branch.spines_volume` — list float'ов (если `calculate_spine_volume`), `spine_utils.py:2202`.
- `branch.spines_obj` — **сбрасывается в None** в `spine_utils.py:2221`; богатые spine-объекты
  строятся ОТДЕЛЬНОЙ поздней стадией, не в `calculate_spines_on_neuron` → воркеру не нужны.

`Limb.spines` (`neuron.py:1040`) и `Neuron.spines` (`neuron.py:3056`) — **read-only
property-агрегаторы** (`nru.feature_list_over_object` обходит дочерние ветки). Сеттера нет;
возвращают `[]`/0, когда у веток `.spines is None`. `n_spines` идёт через
`neuron_utils.py:1032` → `len(obj.spines)` или 0 при None.

**Почему fork читал 0:** классика fork-`Pool` — воркер мутировал `curr_branch.spines` в **своей
форк-копии** нейрона; мутация не вернулась в родитель → ветки остались `spines=None` → `n_spines=0`.
Это НЕ баг хранилища. **Решение Пути A:** воркер **возвращает** `(limb_idx, branch_idx, spines,
spines_volume)`, **родитель переприсваивает** `curr_branch.spines = ...`. In-place мутация в
дочернем процессе не годится.

**Контракт воркера:** вход — ветка (`calculate_spines_on_branch` читает ТОЛЬКО `branch.mesh` и
`branch.skeleton` — суррогат `SimpleNamespace(mesh, skeleton)` достаточен) + per-limb `soma_kdtree`;
выход — `spines` (+ опц. `spines_volume`). spine-меши picklable → возврат через границу процесса ок.
`calculate_spines_on_neuron` зовётся из `neuron.py:2646`.

### 🧱🧱 ВТОРАЯ СТЕНА (2026-05-31): fork → deadlock. Параллелизм ОТЛОЖЕН

Реализовал Путь A (fork-`Pool`, воркер возвращает spines, родитель присваивает; `cgal_folder`
прокинут в `calculate_spines_on_branch` для изоляции CGAL temp-файлов на воркер). Прогон на
fixture-меше:
- **Воркеры зависли в `futex_wait_queue`, 0 CPU**, у каждого **31 поток**. Это **fork-after-threads
  deadlock**: нейрон строится через BLAS/meshparty/CGAL (нативные пулы потоков); `fork()` копирует
  состояние локов, но не потоки-владельцы → первый же `malloc`/BLAS в ребёнке висит вечно.
  (Сразу после `import neurd` потоков 1 — пулы поднимаются именно при **построении** нейрона.)
- Это та же природа, что прошлый «взрыв памяти»: `fork` прогретого многопоточного процесса небезопасен.

**Замеренная развилка (оба плохи под наш критерий «не раздувать RAM»):**
- **`fork` + задушить потоки** (`OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` ДО импорта numpy):
  убирает унаследованные потоки → fork безопасен, RAM низкая (COW). **Цена:** вся декомпозиция
  теряет BLAS-параллелизм — может стать медленнее; нетто-эффект НЕ замерен.
- **`forkserver`/`spawn` + пиклить суррогат ветки:** без deadlock, выход гарантированно идентичен.
  **Цена:** каждый воркер заново импортит neurd (~0.3 ГБ × N) → RAM-пик ~3–3.5 ГБ (≈3× sequential).
  Ещё нужно переинициализировать `*_global` конфиг в воркере (spawn не наследует data_type-настройку).

**РЕШЕНИЕ (пользователь, 2026-05-31): ОТЛОЖИТЬ.** Параллелизм откатан, оставлен только параметр
`cgal_folder` на `calculate_spines_on_branch` (безвреден, пригодится при возврате к теме). Дефолт —
последовательный путь (без изменений). Замечание: целевой H01-меш одно-сомный → на committed
fixture спайнов **0** (sequential тоже даёт 0), так что **correctness параллелизма нельзя
провалидировать локально** — нужен спайн-несущий меш.

**Если возвращаться:** наименее-рискованный замер — вариант «fork + OPENBLAS_NUM_THREADS=1»: одна
env-переменная, COW-дешёвый по RAM, и проверить не стала ли декомпозиция медленнее (нетто).

После #2–#3 (остатки meshlab на соме) пол времени = реальный compute декомпозиции (шипики #4).

---

## 6. GPU — честная оценка (idea пользователя, держим в уме)

GPU помогает там, где **плотная параллельная численная работа**; не помогает там, где оверхед
сабпроцесса (это лечится in-process) или граф-алгоритмы.

- **Single-neuron latency — ограниченный потенциал GPU.** Когда meshlab уйдёт, пол времени —
  скелетонизация (teasar = обход графа, плохо ложится на GPU) и mesh-topology операции. SDF уже
  0.46s. То есть для **одного нейрона** GPU мало что даст после in-process замен.
- **Throughput — вот где GPU реально играет.** `process_all_neurons` гоняет **много** мешей.
  Если сократить RAM на нейрон (#5), можно батчить больше нейронов параллельно, и GPU ускорит
  параллельно-дружелюбные куски (SDF/`ray_trace`, KMeans-сегментация) **через батч**. Поэтому
  **GPU в паре с оптимизацией памяти даёт пропускную способность**, а не скорость одного меша.
- **Что GPU-able по частям:** ray-casting/SDF (open3d RaycastingScene / warp / OptiX),
  KMeans SDF-сегментации (cuML). Скелетонизация и mesh-stitching — нет.
- **Предусловие:** CUDA-Python стек (torch/warp/cupy) не установлен; GPU есть (RTX 5060 = sm_120,
  новая → совместимость колёс проверить). Браться за GPU **после** in-process замен и RAM,
  когда станет ясно, latency или throughput мы упёрлись.

**Итог направления для следующей сессии:** RAM-стена частично снята (live-size 458 → 78 МБ).
Приоритеты по убыванию отдачи:
1. ⛔ **Параллелизация шипиков — ОТЛОЖЕНА** (§5, вторая стена): `fork` → deadlock от унаследованных
   потоков; `forkserver`/`spawn` → RAM ~3×. Блокер `br.spines` снят и контракт ясен, но дешёвой по
   RAM безопасной реализации нет. Если возвращаться — мерить вариант «fork + OPENBLAS_NUM_THREADS=1»
   (нетто-время) на спайн-несущем меше (на committed fixture спайнов 0).
2. **Decimator → open3d** (§5 #2) — output-changing, −12s, судить против Neuron-baseline. **Теперь
   это приоритет #1 из реалистичных.**
3. **Не тратить** время на смену алгоритма скелетонизации (teasar 0.31s) и на GPU (узкое место —
   mesh-топология/граф, не численное; GPU — только throughput после RAM-фикса).
