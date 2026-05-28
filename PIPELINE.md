# NEURD — Конвейер сегментации меша (стадии и швы)

Карта end-to-end pipeline: от меша сегмента до автопруфридинга и
compartment-меток. Стадии и их границы («швы») взяты из исполняемой
спецификации — [tests/integration/test_autoproof_pipeline.py](tests/integration/test_autoproof_pipeline.py)
(каждый `test_N_*` = одна стадия) и оркестратора
[neurd/neuron_pipeline_utils.py](neurd/neuron_pipeline_utils.py).

См. также: [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md) — карта модулей.

---

## Внешние инструменты (важно для запуска)
Mesh-стадии (decimation, soma extraction, decomposition и далее) шеллятся в
`xvfb-run meshlabserver`. Нужны **оба** бинаря на PATH:
- `meshlabserver` (пакет MeshLab) + `xvfb-run` (пакет `xvfb`).
  Поставить: `sudo apt install meshlab xvfb`.

Интеграционный тест помечает mesh-стадии `@_requires_mesh_tools` и **gracefully
скипает** их, если тулинга нет (`shutil.which`). Стадии 0–1 (конфиг, fetch mesh)
работают in-process и проходят без MeshLab.

### ⚠️ Узкие места / кандидаты на замену (для будущей оптимизации, НЕ делать сейчас)
Отмечено по просьбе: места с тяжёлыми/трудно-поддерживаемыми внешними
зависимостями, которые желательно заменить на быстрые in-process аналоги.

| Где | Внешняя зависимость | Через что | Кандидат на замену |
|---|---|---|---|
| Decimation (стадии 2,4) | **MeshLab** (`meshlabserver`+`xvfb`) | `mesh_tools.meshlab.Decimator` | `open3d`/`trimesh` quadric decimation (in-process, без xvfb) |
| Soma extraction / decomposition | **MeshLab** Poisson + interior-removal | `mesh_tools.meshlab.Poisson`, `.remove_interior` | `open3d` Poisson reconstruction (уже в deps) |
| Spine segmentation (стадия 7, ещё не достигнута в прогоне) | **CGAL** | `cgal_Segmentation_Module` в `spine_utils` | проверить, есть ли python-binding/альтернатива |
| (опц.) hole-filling | `pymeshfix` | `pymeshfix_clean` (по умолчанию `False`) | уже опционально, можно не трогать |

> MeshLab — главный тяжёлый внешний инструмент: классический `meshlabserver`
> снят в новых релизах MeshLab (заменён на `pymeshlab`), поэтому это и хрупкая,
> и трудно-поддерживаемая точка. `mesh_tools.meshlab` — upstream-пакет автора,
> замену делать через обёртку в `neurd/`, не правя upstream.

## ⚠️ Известные блокеры выполнения (numpy 2 / trimesh 4 compat)
Конвейер ни разу не прогонялся на numpy≥2 / trimesh≥4 — всплывают compat-баги.
Найдено при первом полном прогоне (2026-05-28):
- **`soma_extraction_utils.py:1002`** — `ordered_mesh_splits = mesh_splits[np.flip(np.argsort(...))]`.
  В trimesh≥4 `mesh.split()` возвращает **list**, а его индексируют numpy-массивом →
  `TypeError: only integer scalar arrays can be converted to a scalar index`.
  Блокирует стадию 4 (soma) и всё ниже. Фикс: `mesh_splits = np.array(mesh.split(...))`
  или индексация через list comprehension. (Вероятно, не единственный такой баг.)

---

## Стадии

Поток данных: `segment_id → mesh → decimated mesh → soma → neuron_obj →
[split] → neuron_obj_axon → neuron_obj_proof → stats`.

| # | Стадия | Вход → выход (шов) | Точка входа | Ключевые модули | MeshLab |
|---|---|---|---|---|---|
| 0 | **Конфиг** | dataset → глобалы модулей | `neurd.set_volume_params(volume)` / `vdi.set_parameters_for_directory_modules()` | `__init__`, `parameter_utils`, `vdi_*` | — |
| 1 | **Fetch mesh** | `segment_id` → `trimesh` | `vdi.fetch_segment_id_mesh(segment_id)` | `volume_utils`, `vdi_microns`/`vdi_h01` | — |
| 2 | **Decimation** | mesh → decimated mesh | `tu.decimate(mesh, ratio)` *(mesh_tools)* | (внешнее, не neurd) | ✅ |
| 3 | **Soma identification** | mesh → soma_products | `sm.soma_indentification(mesh)` | `soma_extraction_utils` | ✅ |
| 4 | **Decomposition** | mesh → `neuron_obj` | `neuron.Neuron(mesh=…).calculate_decomposition_products()` | `neuron`, `preprocess_neuron` | ✅ |
| 5 | **Save / reload** | `neuron_obj` → диск → `neuron_obj` | `vdi.save_neuron_obj` / `vdi.load_neuron_obj` | `vdi_*` | — |
| 6 | **Multi-soma split** | `neuron_obj` → `[neuron_obj, …]` | `neuron_obj.calculate_multi_soma_split_suggestions()` + `.multi_soma_split_execution()` | `soma_splitting_utils`, `proofreading_utils` | ✅* |
| 7 | **Cell type + axon/dendrite** | `neuron_obj` → `neuron_obj_axon` | `npu.cell_type_ax_dendr_stage(n, mesh_decimated)` | см. ниже | ✅* |
| 8 | **Auto proofreading** | `neuron_obj_axon` → `neuron_obj_proof` | `npu.auto_proof_stage(n_axon, mesh_decimated)` | `proofreading_utils`, `error_detection`, `graph_filters` | ✅* |
| 9 | **After-proof stats / compartments** | `neuron_obj_proof` → dict статистик | `npu.after_auto_proof_stats(n_proof)` | `apical_utils`, `synapse_utils`, `neuron_statistics`, `neuron_graph_lite_utils` | — |

\* косвенно: внутри строится/правится меш или вызываются стадии, использующие MeshLab.

---

## Подстадии `cell_type_ax_dendr_stage` (стадия 7)
Из [neuron_pipeline_utils.py](neurd/neuron_pipeline_utils.py):
1. Refine width array — `bu.refine_width_array_to_match_skeletal_coordinates`
2. Branch simplification (≥2 ребёнка) — `nsimp.branching_simplification`
3. (опц.) фильтр low-branch dendrite-кластеров — `pru.apply_proofreading_filters_to_neuron`
4. Match neuron ↔ nucleus — `vdi.nuclei_from_segment_id` + `nru.pair_neuron_obj_to_nuclei`
5. Add synapses — `syu.add_synapses_to_neuron_obj`
6. Spines → head/neck/shaft — `spu.add_head_neck_shaft_spine_objs`
7. Cell typing (E/I) — `ctu.e_i_classification_from_neuron_obj` *(вплетён в pipeline: результат питает axon)*
8. Label axon — `au.complete_axon_processing`
9. Пакетирование статистик — `nst.skeleton_stats_*`, `syu.n_synapses_analysis_axon_dendrite`

## Подстадии `after_auto_proof_stats` (стадия 9)
Neuron stats → synapse stats → cell typing after proof (`ctu`) →
compartment features (`au.axon_features_*`, `apu.compartment_features_*`) →
limb alignment (`apu.limb_features_from_compartment_over_neuron`).

---

## Состояние тестируемости (Фаза 2)
- Швы стадий **уже закодированы** в `test_autoproof_pipeline.py` как
  последовательные `test_N_*` (состояние через `self.__class__`).
- Результат полного прогона (2026-05-28, с установленным `xvfb`, ~48s до блокера):
  **passed** — стадии 0 (конфиг), 1 (fetch mesh), 2 (decimation, MeshLab работает);
  **failed** — стадия 3 (soma) на compat-баге `soma_extraction_utils.py:1002`,
  стадии 4–9 каскадом (нет `neuron_obj`/`n1`/`neuron_obj_axon`).
- Дальнейший прогон блокирует numpy2/trimesh4 compat (см. «Известные блокеры»).
- Характеризационные тесты на отдельных швах (вход→выход стадии) можно добавлять,
  переиспользуя fixture-меш `tests/fixtures/864691135510518224.off` и
  `..._synapses.csv`.

---

## Что делать дальше (для следующих сессий): провести конвейер до конца

Цель — провалидировать сегментацию end-to-end на numpy≥2 / trimesh≥4, починив
compat-баги. Это **точечные багфиксы**, НЕ замена бинарников (та — отдельная
оптимизация, см. таблицу «кандидаты на замену» выше; сейчас не трогаем).

**Предпосылки окружения:** `meshlab` + `xvfb` установлены; conda-env `neurd`
(`source ~/miniforge3/etc/profile.d/conda.sh && conda activate neurd`).

**Итеративный цикл (run → fix → run):**
1. Прогнать интеграционный тест на fixture-меше:
   ```
   python -m pytest tests/integration/test_autoproof_pipeline.py -p no:cacheprovider -v
   ```
   (полный прогон медленный; до текущего блокера ~48s)
2. Найти **первый реальный** traceback (не каскадные `AttributeError`, которые
   возникают из-за того, что предыдущая стадия не записала `self.__class__.*`).
3. Починить причину. Ожидаемый класс багов — numpy 2 / trimesh 4 несовместимости:
   - индексация Python-list numpy-массивом (trimesh `.split()` теперь отдаёт list);
   - удалённые алиасы numpy (`np.float_`/`np.int_` — частично закрыты шимом в `__init__`);
   - изменения API trimesh 3→4.
   Фикс — в `neurd/` (не в upstream-пакетах). По возможности добавить регресс-точку.
4. Повторять, пока не пройдут все 9 стадий (`test_1` … `test_9`).

**Известный первый блокер:** `soma_extraction_utils.py:1002` —
`mesh_splits[np.flip(np.argsort(...))]`, где `mesh_splits` = list (trimesh≥4).
Фикс: обернуть в `np.array(...)` или индексировать через list comprehension.

**Прогресс на 2026-05-28:** стадии 0–2 (config, fetch mesh, decimation) — PASS;
стадия 3 (soma) — упирается в баг выше; 4–9 ещё не достигнуты.

После того как конвейер пройдёт целиком — зафиксировать стадии характеризационными
тестами на швах (вход→выход) и только потом рассматривать замену MeshLab/CGAL.
