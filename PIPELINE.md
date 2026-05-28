# NEURD — Конвейер сегментации меша (стадии и швы)

Операционная карта end-to-end pipeline: от меша сегмента до автопруфридинга и
compartment-меток. Стадии и «швы» взяты из исполняемой спецификации
[tests/integration/test_autoproof_pipeline.py](tests/integration/test_autoproof_pipeline.py)
(каждый `test_N_*` = одна стадия) и оркестратора
[neuron_pipeline_utils.py](neurd/neuron_pipeline_utils.py).

См. также:
- [PIPELINE_GEOMETRY.md](PIPELINE_GEOMETRY.md) — **как пайплайн работает геометрически и в коде** (что и почему можно/нельзя менять).
- [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md) — карта модулей.

## Статус: ПАЙПЛАЙН ЗЕЛЁНЫЙ (2026-05-28)
Все 9 стадий проходят end-to-end на numpy 2 / trimesh 4, **без Docker**. Юнит+оракул: 84 passed, 1 skipped.

**Окружение:** conda-env `neurd` (`source ~/miniforge3/etc/profile.d/conda.sh && conda activate neurd`).
Mesh-стадии шеллятся в `xvfb-run meshlabserver` — нужны оба бинаря (`sudo apt install meshlab xvfb`).
Интеграционный тест помечает mesh-стадии `@_requires_mesh_tools` и скипает их, если тулинга нет.

**Прогон:** `python -m pytest tests/integration/test_autoproof_pipeline.py -p no:cacheprovider -v` (~4–5 мин).

---

## Стадии

Поток данных: `segment_id → mesh → decimated mesh → soma → neuron_obj →
[split] → neuron_obj_axon → neuron_obj_proof → stats`.

| # | Стадия | Вход → выход (шов) | Точка входа | Ключевые модули | MeshLab |
|---|---|---|---|---|---|
| 0 | **Конфиг** | dataset → глобалы модулей | `neurd.set_volume_params(volume)` | `__init__`, `parameter_utils`, `vdi_*` | — |
| 1 | **Fetch mesh** | `segment_id` → `trimesh` | `vdi.fetch_segment_id_mesh(segment_id)` | `volume_utils`, `vdi_microns`/`vdi_h01` | — |
| 2 | **Decimation** | mesh → decimated mesh | `tu.decimate(mesh, ratio)` *(mesh_tools)* | (внешнее) | ✅ |
| 3 | **Soma identification** | mesh → soma_products | `sm.soma_indentification(mesh)` | `soma_extraction_utils` | ✅ |
| 4 | **Decomposition** | mesh → `neuron_obj` | `neuron.Neuron(mesh=…).calculate_decomposition_products()` | `neuron`, `preprocess_neuron` | ✅ |
| 5 | **Save / reload** | `neuron_obj` → диск → `neuron_obj` | `vdi.save_neuron_obj` / `vdi.load_neuron_obj` | `vdi_*` | — |
| 6 | **Multi-soma split** | `neuron_obj` → `[neuron_obj, …]` | `neuron_obj.calculate_multi_soma_split_suggestions()` + `.multi_soma_split_execution()` | `soma_splitting_utils`, `proofreading_utils` | ✅* |
| 7 | **Cell type + axon/dendrite** | `neuron_obj` → `neuron_obj_axon` | `npu.cell_type_ax_dendr_stage(n, mesh_decimated)` | см. ниже | ✅* |
| 8 | **Auto proofreading** | `neuron_obj_axon` → `neuron_obj_proof` | `npu.auto_proof_stage(n_axon, mesh_decimated)` | `proofreading_utils`, `error_detection`, `graph_filters` | ✅* |
| 9 | **After-proof stats / compartments** | `neuron_obj_proof` → dict статистик | `npu.after_auto_proof_stats(n_proof)` | `apical_utils`, `synapse_utils`, `neuron_statistics` | — |

\* косвенно: внутри строится/правится меш или вызываются стадии, использующие MeshLab.

### Подстадии `cell_type_ax_dendr_stage` (стадия 7)
1. Refine width array — `bu.refine_width_array_to_match_skeletal_coordinates`
2. Branch simplification (≥2 ребёнка) — `nsimp.branching_simplification`
3. (опц.) фильтр low-branch dendrite-кластеров — `pru.apply_proofreading_filters_to_neuron`
4. Match neuron ↔ nucleus — `nru.pair_neuron_obj_to_nuclei`
5. Add synapses — `syu.add_synapses_to_neuron_obj`
6. Spines → head/neck/shaft — `spu.add_head_neck_shaft_spine_objs`
7. Cell typing (E/I) — `ctu.e_i_classification_from_neuron_obj` *(результат питает axon)*
8. Label axon — `au.complete_axon_processing`
9. Пакетирование статистик — `nst.skeleton_stats_*`, `syu.n_synapses_analysis_axon_dendrite`

### Подстадии `after_auto_proof_stats` (стадия 9)
Neuron stats → synapse stats → cell typing after proof → compartment features
(`au.axon_features_*`, `apu.compartment_features_*`) → limb alignment.

---

## Внешние инструменты и кандидаты на замену

Тяжёлые/хрупкие внешние зависимости. **MeshLab — главная:** классический `meshlabserver`
снят в новых релизах (заменён на PyMeshLab), это источник целого класса version-багов
(см. Баги 2/4 ниже). `mesh_tools.*` — upstream-пакеты автора, замену делать обёрткой/патчем
в `neurd/`, не правя upstream.

| Где | Зависимость | Через что | Замена |
|---|---|---|---|
| Decimation (стадии 2,4) | MeshLab | `meshlab.Decimator` | `open3d`/`trimesh` quadric decimation |
| Soma/decomp Poisson | MeshLab | `meshlab.Poisson` | `open3d` Screened Poisson |
| Hole filling | MeshLab | `meshlab.FillHoles` | `trimesh.fill_holes` / open3d |
| Interior removal | MeshLab | `meshlab.Interior` (Ambient Occlusion) | **нет прямого аналога** — узкое место |
| Soma/spine segmentation | CGAL `cgal_Segmentation_Module` | `tu.mesh_segmentation()` | **СДЕЛАНО** — питон-stub (Баг 3) |
| Скелетонизация | meshparty / CGAL | `preprocess_neuron` | отдельная зависимость, НЕ meshlab |

> Полностью убрать `meshlabserver`+`xvfb` = заменить 4 meshlab-операции (Decimator,
> Poisson, FillHoles, Interior). Первые три — лёгкие in-process аналоги; Interior — сложный.
> Делать только на зелёном пайплайне + характеризационных тестах (см. ниже).

---

## Применённые compat-патчи (numpy 2 / trimesh 4 / мёртвый meshlabserver)

Все патчи — в `neurd/` (не в upstream). Большинство — monkeypatch'и в
[neurd/__init__.py](neurd/__init__.py), применяемые при `import neurd` (до импорта mesh_tools).

| # | Симптом | Причина | Фикс (где) |
|---|---|---|---|
| 1 | `TypeError ... scalar index` в soma split | trimesh≥4 `mesh.split()` → `list`, не `ndarray` | `soma_extraction_utils.py` (72,1002,1123): `list(...)` + list-comprehension |
| 2 | Poisson не выполняется, нет выходного файла | `meshlab.Poisson` пишет `<xmlfilter>` XML, MeshLabServer 2020.09 его игнорирует | `__init__.py`: патч `Poisson.initialize_script_filters` → `type=Rich*` (формат `<filter>`) |
| 3 | `NameError: csm` (CGAL не установлен) | C++ расширение CGAL отсутствует | `__init__.py`: stub `_cgal_segmentation.py` в `sys.modules['cgal_Segmentation_Module']` (ray_trace SDF + KMeans) |
| 4 | `scipy ValueError: axis 0 index ... exceeds` | MeshLabServer 2020.09 OFF-экспортёр: компактные вершины, но грани в старой нумерации | `__init__.py`: патч `Meshlab.fetch_mesh_from_off` → перенумерация `searchsorted(unique(faces), faces)` |
| 5 | `IndexError ... size 1` в multi-soma split | две сомы на одном стартовом узле лимба → путь из 1 узла | `proofreading_utils.py:1147`: guard `if len(soma_to_soma_path) < 2: break` |
| 6 | `ValueError: data_pts ... 2 dimensions` | новая pykdtree требует 2D, upstream строит KDTree из 1D дистанций | `__init__.py`: обёртка `skeleton_utils.KDTree` (1D→(N,1)) |
| 7 | `numpy_dep has no attribute 'in1d'` | `np.in1d` удалён в numpy 2 | `__init__.py`: `numpy.in1d = isin` + `numpy_dep.in1d = isin` |
| 8 | стадия 9: dotmotif | autoproof обязателен dotmotif; mainline — другой грамматич. диалект | форк reimerlab (см. ниже) + stub `dotmotif.executors.Neo4jExecutor` в `__init__.py` |

> Детали корней любого бага — в `git log` соответствующих правок и в memory-записях
> сессии 2026-05-28. Здесь — только что/где, чтобы понимать назначение патчей в `__init__.py`.

### Новые зависимости (стадия 9 / autoproof)
```
pip install --no-deps git+https://github.com/reimerlab/dotmotif
pip install grandiso lark-parser
```
**Форк reimerlab, не mainline PyPI** (у mainline другой синтаксис мотивов, спотыкается на
`[sk_angle <= ...]`). neo4j-хвост (py2neo/tamarind/dask/neuprint) НЕ нужен — обрублен стабом
`Neo4jExecutor`. ⚠️ Ещё не внесено в `requirements.txt` (git-форк требует особого оформления).

---

## Тестируемость
- Швы стадий закодированы в `test_autoproof_pipeline.py` как `test_N_*` (состояние через `self.__class__`).
- **Оракул-тест CGAL-шва**: [tests/integration/test_cgal_segmentation_oracle.py](tests/integration/test_cgal_segmentation_oracle.py).
  Эталон оригинального CGAL закоммичен: `tests/990_mesh.off` (7240 граней) +
  `990_mesh-cgal_3_0.20.csv` (кластер/грань) + `..._sdf.csv`. Три теста:
  целостность эталонов (всегда), контракт провайдера (длины, int-кластеры, SDF∈[0,1]),
  fidelity (корреляция SDF с эталоном ≥0.85; питон-stub даёт ~0.95). Скипается без провайдера.
- Характеризационные тесты на других швах можно добавлять тем же приёмом
  (вход→выход стадии, эталон на fixture-меше). Делать ДО замены meshlab.

---

## Производительность: гипотезы (НЕ профилировано)

> ⚠️ Статический анализ, не профайлер. Перед оптимизацией — подтвердить
> `cProfile`/`memray` на fixture-меше. Профиль наблюдался ~8 ч + 32 ГБ на 0.5 ГБ меш.

1. **MeshLab через диск+subprocess** (decimate/poisson/interior ×N): ASCII-сериализация
   огромного меша + спавн процесса. Устранимо — in-process open3d/trimesh.
2. **Eager deepcopy submesh на каждый Branch** (`neuron.py` `Branch.__init__`): тысячи
   материализованных submesh-копий → главный драйвер RAM. Частично устранимо (view вместо copy).
3. **deepcopy целого Neuron** на границах стадий (`return_copy=True`): удваивает пик RAM.
4. **Скелетонизация** (meshparty/CGAL): легитимно дорогая CPU-работа, переписыванием не убрать.
5. **Пер-branch циклы** (ширины/статистики/graph-queries): частично векторизуемо/кэшируемо.

#1 и #2 — «low-hanging fruit», структурно устранимы. НО: #2/#3 трогают `Branch`/`Neuron`
(сердце core clump) — опасно без характеризационных тестов.

---

## Что делать дальше
1. **Зафиксировать новые зависимости** (dotmotif git-форк + grandiso + lark-parser) — решить, как оформить git-форк в requirements.
2. **Характеризационные тесты на швах** стадий (приём с оракулом), пока пайплайн зелёный.
3. **Замена `meshlabserver`** на in-process (open3d/trimesh) — 4 операции, Interior сложнейшая;
   фулл-цель: убрать meshlabserver+xvfb. Только после п.2.
4. Отложено: decouple `cell_type_utils`, `DictType→dict` (детали — в memory).
