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

## 0. ОБНОВЛЕНИЕ 2026-05-31: смена парадигмы + реальные H01-нейроны

Сессия вышла за рамки «оптимизации» и вскрыла **две Docker-fidelity регрессии** форка
(уход от Docker-тулчейна сломал реальные вещи). Проверено на настоящих H01-нейронах
из `Diplom/notebooks/notebooks/H01/*.off` через новый бенчмарк.

### Инструмент: [tests/tools/benchmark_neuron.py](tests/tools/benchmark_neuron.py)
Переиспользуемый time+memory бенчмарк, **точно повторяет** `process_all_neurons` (h01,
`load → Neuron → save_segmentation`). Даёт: per-stage wall-clock, RSS-таймлайн + пик
(`ru_maxrss`), cProfile (cumulative/tottime/**junk по ncalls**), опц. tracemalloc. Пишет
отчёт + `profile.prof`. Флаги: `--no-profile` (чистый wall), `--no-spines`, `--no-save`,
`--tracemalloc`. **Пишет отчёт даже при краше пайплайна.** Запуск:
`python tests/tools/benchmark_neuron.py --neuron <path.off> --out <dir>`.

### Регрессия №1: Poisson (no-op → краш на сложных H01)
Форк заменил MeshLab Poisson на **no-op** (вывод: «локальный meshlab 2020.09 его не умеет,
значит Poisson не нужен»). Но это вывод из **сломанного локального meshlab + microns-фикстуры,
которой Poisson не нужен**. В Docker Poisson **работал**: чинил EM-меши (щели/само-контакты) в
связную поверхность. Без него у сложных нейронов лимб-submesh распадается → краш в
`preprocess_neuron.py:209 correspondence_1_to_1` («not just one mesh»). Не регрессия именно
`optimize_segmentation` — `main` упал бы так же. Пример: `neuron_1830470325.off` (3.27М граней)
падал; Docker давал 4 лимба + 214 спайнов (`H01_Seg/neuron_1830470325/`).

**Фикс (за флагом `NEURD_REAL_POISSON=1`, дефолт = no-op):** [_mesh_ops.py](neurd/_mesh_ops.py)
`poisson_surface_reconstruction_meshlab` — настоящий MeshLab Screened Poisson через **pymeshlab**
(in-process, без Docker), те же параметры (depth=11, fulldepth=6, pointweight=4, samplespernode=1.5,
scale=1.1, iters=8), что форк удалил. Подключён в [__init__.py](neurd/__init__.py) под флагом.
- ✅ **краш чинит** (нейрон проходит).
- ⚠️ но **реконструкция ещё не пиксель-в-пиксель Docker**: 11 лимбов vs Docker 4 (open3d-версия
  давала 18 — хуже: она сабсэмплит точки и рвёт тонкие отростки). Тюнинг depth/params **отложен**:
  параметры уже = Docker (см. ниже), а за Docker-точностью разбиения мы **решили не гнаться**
  (пользователь: «такое разбиение пойдёт», 11 лимбов ок). 4 лимба Docker = 2 крупных дерева
  (82+85 веток) + 2 огрызка; наши 11 = те же 2 дерева, распавшиеся на ~9 кусков. Лимбы режутся из
  **исходного** меша минус грани сомы (`_segment_limbs_from_soma`, [preprocess_neuron.py:2736](neurd/preprocess_neuron.py#L2736)),
  Poisson влияет на их число лишь косвенно (через какие грани = сома).
- 🔴 **РЕГРЕССИЯ ВРЕМЕНИ 55→77 мин — это и есть «замена Poisson».** Большой `1830470325`:
  open3d-Poisson (лёгкий) → **53 мин / 18 лимбов**; pymeshlab MeshLab-Poisson depth=11 (тяжёлый,
  Docker-верный) → **77 мин / 11 лимбов**. То есть +24 мин куплены за лучшую (но всё ещё не
  Docker) связность. depth=11 на 3.27М граней = **766s чистого CPU за 17 вызовов** — крупнейший
  одиночный расход (см. §0 «Большой нейрон»). Параметры pymeshlab сверены 1:1 с Docker-скриптом
  (`mesh_tools/meshlab.py:486` Screened Poisson): depth=11/fulldepth=6/cgDepth=0/scale=1.1/
  samplesPerNode=1.5/pointWeight=4/iters=8 — крутить «ближе к Docker» нечего.

### Регрессия №2: CGAL→KMeans → СПАЙНЫ СЛОМАНЫ ВЕЗДЕ
`_cgal_segmentation.py` (наш Python-стенд) заменил CGAL C++ SDF-segmentation на **KMeans по 1D-SDF**.
Для **сомы** (грубый контраст толстое/тонкое) ок. Для **спайнов** нужна тонкая over-сегментация —
KMeans её не даёт → **0 спайнов на ВСЕХ нейронах** (microns-fixture, малый и большой H01; Docker
давал 214). Мои §2-оптимизации (KMeans n_init, lazy-shaft) **ускоряли стадию, выдающую ноль**.

### РЕШЕНИЕ (пользователь): ОТКАЗ ОТ ШИПИКОВ
`process_all_neurons.py`: `Neuron(..., calculate_spines=False)` в обоих местах + убрано сохранение
спайн-мешей в `save_segmentation` (limb/branch меши+скелеты сохраняются как раньше). Эффект на
малом H01: **215s → 123s (−40%)**, спайн-папок 0, limb/branch с widths на месте (протестировано).
Стадия шипиков (сломана + дорога) выпилена целиком — §2/§4a про неё теперь моот.

### Профиль МАЛОГО H01 (no-spines) — где время РЕАЛЬНО
| Операция | ~время | природа |
|---|---|---|
| ✅ **Объём сомы** — **УДЕШЕВЛЁН** (коммит `7918ffc`) | **0s** (было ~46s) | см. ниже |
| **Meshlab-сабпроцессы** (`remove_interior` + децимация, 5 спавнов xvfb) | ~19s | §5 #3 |
| networkx-граф (correspondence `bfs_edges` ×431К) | ~12s | граф-обвязка |
| Скелетонизация (meshparty, в потоках) | скромно | НЕ доминанта |

✅ **СДЕЛАНО — объём сомы (`7918ffc`, −35s wall на малом H01: 146.8→112.2s, вывод побайтово
тот же 5 лимбов/28 веток):** `soma_volumes` ([preprocess_neuron.py:2916](neurd/preprocess_neuron.py#L2916))
питает только `Soma.volume` (стат суммарного объёма + `Soma.__eq__`), НЕ декомпозицию. Дефолтный
`mesh_volume` watertight-ит через `fill_mesh_holes_with_fan` (~46s на большой соме) и **сам падает
на convex_hull**, если fan не сомкнул — поэтому берём `mesh_volume(..., watertight_method="convex_hull")`
напрямую. `fill_mesh_holes_with_fan` исчез из профиля на обоих нейронах.

**Выводы по малой основной фазе:**
- **GPU не поможет** — топ-расходы это mesh-починка/сабпроцессы/графы, не числодробление. SDF уже 0.46s.
- **Kimimaro почти не поможет** — скелетонизация-ядро скромно (на потоках).
- **#1 оставшийся рычаг:** `remove_interior` → in-process через pymeshlab (−~19s сабпроцессов).

### ⭐ Большой нейрон `1830470325` (3.27М граней): где РЕАЛЬНО время
Прогон no-spines + real Poisson + удешевлённая сома (`/tmp/bench_big_nospines`): **4515s ≈ 75 мин**,
12 лимбов / 220 веток / 0 спайнов, RAM пик **8.27 ГБ**. cProfile: total 5301s.

**Сравнение прогонов большого (объясняет регрессию 55→77 и роль шипиков):**
| Конфиг | Poisson | Лимбы | Время | вывод |
|---|---|---|---|---|
| open3d Poisson + spines | лёгкий | 18 | **53 мин** | «оригинальные 55 мин» |
| pymeshlab depth=11 + spines | тяжёлый | 11 | **77 мин** | замена Poisson = +24 мин |
| pymeshlab depth=11, **no-spines**, cheap-soma | тяжёлый | 12 | **75 мин** | −спайны дали лишь ~110s! |

👉 **На большом нейроне шипики НЕ были бочтлнеком** (−110s), в отличие от малого (−368s). Большой
ограничен Poisson + meshlab-сабпроцессами + геометрией декомпозиции — они есть в обоих прогонах.

**Чистые self-цифры (где CPU реально горит):**
| Реальная работа | CPU self | вызовов | что |
|---|---|---|---|
| **Real Poisson (pymeshlab)** | **766s** | 17 | depth=11; крупнейший расход; **мы сами добавили ради краш-фикса** |
| meshlab-сабпроцессы | ~346s | — | Decimator 146 + Interior 100 + FillHoles 100 (xvfb/meshlabserver) |
| `closest_distance_between_meshes` | 166s | 10731 | геометрия привязки floating-кусков/лимбов |
| `signed_distance` (embree) | 71s | 773 | лучевые запросы |

### ⚠️ Как читать «3636s acquire of _thread.lock» (НЕ баг, НЕ отдельное время)
Топ профиля по tottime — `{method 'acquire' of '_thread.lock'}` 3636s. Прослежена цепочка:
`lock.acquire ← Condition.wait ← Event.wait ← главный поток`. Это **главный (единственный
профилируемый) поток Python, заблокированный в ожидании**, пока работа идёт ВНЕ Python:
(1) в нативном C, отпускающем GIL — pymeshlab Poisson, embree (`signed_distance`); (2) во внешних
процессах — `meshlabserver`/xvfb через `subprocess.communicate`. cProfile видит только Python-байткод,
поэтому «спящее» время пишет как ожидание лока. **Эти 3636s НЕ складываются** с self-цифрами выше —
это те же секунды стены, вид со стороны Python. Исчезнут только вместе с нативной работой под ними.

### ⭐ Poisson depth-sweep (2026-05-31) — depth=8 новая рабочая точка

Почему depth дорог: цена Poisson определяется **глубиной октодерева (разрешением 2^depth)**, а не
числом граней. depth=11 = сетка 2048³ — восстанавливает тонкие детали; для гладкого толстого блоба
сомы это оверкилл. **Сома = НЕ «самая большая компонента»** (весь нейрон — один связный меш); она
выделяется классификацией **толщины каждой грани** (SDF), а SDF требует замкнутой поверхности →
Poisson. Поэтому depth бьёт по детекции сомы.

Env-регуляторы (дефолты = без изменений): `NEURD_POISSON_DEPTH` (11), `NEURD_POISSON_ITERS` (8),
`NEURD_POISSON_BACKEND` ("meshlab" или "open3d"), `NEURD_SOMA_OUTER_DECIM` / `NEURD_SOMA_INNER_DECIM` (0.25).

| Конфиг | Время | Лимбы | Ветки | RAM | Сома |
|---|---|---|---|---|---|
| depth=11 pymeshlab (Docker) | 4515s (75м) | 12 | 220 | 8.27 ГБ | ✅ |
| depth=9 pymeshlab | 1589s (26.5м) | 5 | 220 | 7.8 ГБ | ✅ |
| **depth=8 pymeshlab** ⭐ | **1155s (19.2м)** | **5** | **196** | **8.36 ГБ** | ✅ |
| depth=7 pymeshlab | 179s† | — | — | 2.95 ГБ | ❌ 0 сом «No Somas» |
| open3d depth=9 | 861s† | — | — | 4.2 ГБ | ❌ «not just one mesh» |
| depth=9 + decim0.15 + iters6 | 1397s† | — | — | 4.8 ГБ | ❌ «not just one mesh» |
| *Docker* | — | *4* | — | — | ✅ |

† умер на полпути, не досчитал.

**Выводы:**
- ⭐ **depth=8 — новая лучшая точка: −27% от depth=9 (19.2 vs 26.5 мин), те же 5 лимбов, сома
  детектируется.** Ветки 196 vs 220 — чуть грубее, принято (качество сомы/разбиения не критично).
- **depth=9 — тоже работает** (fallback если depth=8 ломает другие нейроны).
- **depth=7 = ниже порога детекции сомы** → 0 сом → краш. depth=8 — у нового «пола».
- **open3d depth=9 → краш** (`correspondence_1_to_1`): open3d при depth<11 теряет тонкие отростки →
  рвёт меш. open3d depth=11 работал (53 мин / 18 лимбов), но медленнее depth=8 pymeshlab. open3d-путь
  закрыт — pymeshlab точнее реконструирует EM-поверхность.
- 🔴 **Агрессивная децимация соме-меша КОНТРПРОДУКТИВНА (опровергнута замером).** Идея: грубее меш →
  дешевле весь соме-этап. Реально: соме-экстракция ретраит (`max_fail_loops`) на борделайн-кандидатах
  → Poisson-вызовов стало **11 вместо 6**, времени не сэкономлено, И вернулся исходный краш
  `correspondence_1_to_1 "not just one mesh"` (огрубление < порога связности).

**Рычаги для большого ПОСЛЕ depth=8 (Poisson выжат, дальше НЕ сома):**
1. ⭐ **meshlab-сабпроцессы** (Decimator + Interior + FillHoles) — перенести в in-process pymeshlab.
   **Не трогает реконструкцию → низкий риск краша.** Теперь это рычаг №1.
2. `closest_distance_between_meshes` — геометрия привязки floating-кусков (сложнее, output-changing).
3. ✅ **depth=8 — зафиксировать дефолтом** (через `NEURD_POISSON_DEPTH=8` в process_all_neurons).
   Пока не закоммичено.

### RAM на реальных H01
Малый (507К граней): пик ~1.65 ГБ (после удешевления сомы; было ~1.8). Большой `1830470325`
(3.27М граней): пик **~8.36 ГБ** (depth=8). Децимация уже идёт внутри соме-экстракции (0.25×0.25) — НЕ рычаг.

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

### 4b. Полный профиль СБОРКИ после спайн-фиксов (2026-05-31, fixture, cProfile)

После KMeans+lazy-shaft ландшафт **сместился** — шипики больше не доминанта. Реальная сборка
**92s** (cProfile раздут до 139s; пропорции верны):

| Стадия | cumtime (проф.) | Что внутри |
|---|---|---|
| **`preprocess_limb` — скелетонизация+обвязка** | **~84s (доминанта)** | `skeletonize_and_clean_connected_branch_CGAL` 52s (3×); `convert_skeleton_to_graph` 16s (**1417 вызовов ≈29/ветку**); `skeleton_obj_to_branches` 12.7s; `filter_limb_correspondence_for_end_nodes` 11s; `resolve_empty_conflicting_face_labels` 14s |
| **Сома-экстракция** | ~34s | meshlab `remove_interior` 11.6s (subprocess ×4), poisson-watertight чеки, `cgal_segmentation` (уже с KMeans-фиксом) |
| Шипики | 14.3s | ✅ оптимизированы |

**Подтверждает §4:** teasar-ядро дёшево, дорога **обвязка скелета — графы/networkx**. Самый горячий
self-time: **`networkx.add_edges_from` 9.7s self (2775 вызовов)** внутри `convert_skeleton_to_graph`.

⚠️ **Эти выигрыши КАЧЕСТВЕННО сложнее спайн-фиксов:**
- Граф-обвязка (`convert_skeleton_to_graph`/`add_edges_from`) — **upstream `mesh_tools/skeleton_utils`**,
  не наш код; networkx-построение графа. Ускорить можно (scipy-sparse/igraph вместо networkx, или
  кэш если 1417 вызовов редундантны — НЕ проверено), но **output-риск** (топология скелета) и upstream.
- `remove_interior` (11.6s, meshlab subprocess) — §5 #3, нужен in-process raycasting (open3d
  RaycastingScene), высокая сложность, output-changing.

**Развилка для след. сессии:** лёгкие высоко-уверенные спайн-выигрыши исчерпаны. Дальше — либо
браться за skeleton-граф-обвязку (замерить редундантность `convert_skeleton_to_graph`: если граф
строится повторно на одном скелете — кэш дешёв и output-preserving; если структурно — дорого/рискованно),
либо за `remove_interior` (§5 #3). Оба — не «лёгкие места».

---

## 5. Plan вперёд (по убыванию замеренной отдачи) + СТЕНА

> ⭐ **АКТУАЛЬНЫЕ приоритеты (2026-05-31) — см. §0 «Poisson depth-sweep».** Большой H01:
> **depth=9 ПРОВЕРЕН** — 75м→26.5м (−2.84×), Poisson 766→167s, ближе к Docker (5 лимбов vs 4).
> Poisson/сома-рычаг **исчерпан** (depth<9 и децимация ломают пайплайн — замерено). Дальше:
> **(1) meshlab Decimator/Interior/FillHoles → in-process pymeshlab (−262s, низкий риск)**,
> (2) `closest_distance` геометрия (151s). depth=9 зафиксировать дефолтом (пользователь думает).
> Таблица ниже — история малого нейрона/fixture; для большого см. §0.

| # | Цель | Где | Сложность | Статус/ожидание |
|---|---|---|---|---|
| 1 | ~~fill_holes → no-op~~ | `__init__.py` | — | ✅ **СДЕЛАНО** (`5aca8c0`) |
| 2 | **Decimator → open3d/pymeshlab** | патч `__init__.py` | низко (fidelity ✓) | **держим** — output-changing: меняет сому (median 0.533→0.479), −12s на малом / **−146s на большом**. Прототип был, откатан. Судить против baseline |
| 3 | **remove_interior** (meshlab `Interior`) | соме-стадия | высоко | реальная операция (не no-op!) — in-process через pymeshlab/рейкастинг (open3d RaycastingScene). **−100s на большом** |
| 4 | ~~**Объём сомы (fill_mesh_holes_with_fan)**~~ | `preprocess_neuron.py:2916` | — | ✅ **СДЕЛАНО** (`7918ffc`) — convex_hull, −35s малый |
| 5 | ~~**Шипики — параллелизация**~~ | — | — | ⛔ **ЗАБРОШЕНО:** шипики выпилены целиком (`calculate_spines=False`, см. §0). Стенки про fork/deadlock ниже — историческая справка |
| 6 | ~~**RAM** (deepcopy/кэши)~~ | `neuron.py` | — | ✅ **СДЕЛАНО** (`fedde35`, `2a86407`, `3643d3e`) — live-size 458 МБ → 78 МБ |

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
