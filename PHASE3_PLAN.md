# Фаза 3 — Освобождение от `celiib/mesh_tools:v4` и локальный запуск

Документ для следующих сессий: что менять, в каком порядке, какие тесты писать.

---

## Контекст и цель

Сейчас тесты запускаются только внутри Docker-контейнера на базе образа `celiib/mesh_tools:v4`
(Python 3.8, автора NEURD). Это создаёт два неудобства:
- **Привязка к чужому образу** — он может исчезнуть или не обновляться.
- **Нельзя запускать тесты локально** — разработчик вынужден держать Docker.

**Цель:** запускать тесты и сам pipeline напрямую в virtualenv на Python 3.12, без Docker.

---

## Что выяснено в результате анализа (2026-05-27)

### Локальная среда уже подходит
- Python **3.12.3** установлен локально — лучше 3.11.
- Все нужные пакеты есть на PyPI для Python 3.12:
  - `open3d==0.19.0` (вместо устаревшего `==0.11.2`)
  - `meshparty==2.0.3`
  - `trimesh>=4`
  - `pykdtree==1.4.3`

### `mesh_processing_tools` — чистый Python
Wheel `mesh_processing_tools-1.0.4-py3-none-any.whl` — **без C-extensions**.
Устанавливается на любой Python через `--no-deps`, проблема только в задекларированном
`open3d==0.11.2`. При `--no-deps` это ограничение обходится.

### Два реальных блокера

| Блокер | Файл | Симптом | Решение |
|--------|------|---------|---------|
| `numpy.float_` удалён в numpy 2 | `datasci_tools/numpy_dep.py` | `AttributeError` при любом `import datasci_tools` | monkeypatch в `neurd/__init__.py` |
| `import ipyvolume as ipv` на верхнем уровне | `mesh_tools/skeleton_utils.py` | `ModuleNotFoundError` при `from mesh_tools import skeleton_utils` | stub-модуль в `neurd/__init__.py` |

`open3d==0.11.2` уже **не блокер** — на PyPI есть `open3d==0.19.0` для Python 3.12.

---

## Шаги (в порядке выполнения)

### Шаг 1 — numpy compatibility shim (30 мин)

Добавить в начало `neurd/__init__.py` до любых других импортов:

```python
# Compatibility shim: datasci_tools uses np.float_ which was removed in numpy 2.
import numpy as _np
if not hasattr(_np, 'float_'):
    _np.float_ = _np.float64
    _np.int_ = _np.int64
    _np.complex_ = _np.complex128
    _np.bool_ = _np.bool_   # уже есть, но на всякий случай
del _np
```

**Почему безопасно:** `np.float_` — это просто алиас для `np.float64`, который никуда
не делся. Monkeypatch не меняет поведение — только восстанавливает удалённый алиас.

**Тест:** `python3 -c "import neurd"` должен завершиться без ошибок.

---

### Шаг 2 — ipyvolume / cloudvolume stubs (30 мин)

`mesh_tools/skeleton_utils.py` делает `import ipyvolume as ipv` на верхнем уровне.
Мы не контролируем этот пакет, но можем вставить stub в `sys.modules` **до** того,
как neurd импортирует mesh_tools:

Добавить в `neurd/__init__.py` (после шага 1, до остальных импортов):

```python
# Inject stubs for optional mesh_tools dependencies so that skeleton_utils
# can be imported without ipyvolume / cloud-volume installed.
import sys as _sys
from types import ModuleType as _ModuleType

def _make_stub(name: str) -> _ModuleType:
    stub = _ModuleType(name)
    stub.__getattr__ = lambda self, n: None   # type: ignore[method-assign]
    return stub

for _pkg in ('ipyvolume', 'cloudvolume'):
    if _pkg not in _sys.modules:
        _sys.modules[_pkg] = _make_stub(_pkg)
del _sys, _ModuleType, _make_stub, _pkg
```

**Почему безопасно:** stub подставляется только если пакет **не установлен**.
Если `ipyvolume` установлен (например, для визуализации) — импортируется настоящий.

**Тест:** `python3 -c "from mesh_tools import skeleton_utils"` без ipyvolume.

---

### Шаг 3 — Локальная установка в virtualenv (1-2 ч)

Создать `requirements-local.txt` (или просто инструкцию в README):

```
# Ядро pipeline
numpy>=1.24,<3       # <3 на случай будущих изломов; shim покрывает 2.x
scipy
pandas>=2.0
networkx>=3
matplotlib>=3.7
trimesh>=4
meshparty>=2.0
scikit-learn>=1.3
h5py
tqdm

# Авторские пакеты (без ограничений по версии — ставим что есть на PyPI)
datasci-stdlib-tools>=1.0.1
machine-learning-tools>=1.0.0
graph-nx-tools>=1.0.0
neuron_morphology_tools>=1.0.1
code_structure_tools>=1.0.2

# mesh_tools без пина на open3d==0.11.2
# (ставим --no-deps, потом зависимости вручную)
# см. install.sh ниже

# Mesh pipeline
open3d>=0.19          # Python 3.12 wheel есть на PyPI
pykdtree>=1.4         # нужен mesh_tools внутренне
pymeshfix>=0.16       # нужен mesh_tools/meshlab
```

Скрипт установки `scripts/install_local.sh`:

```bash
#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate

pip install uv

# mesh_processing_tools ставим без deps (обходим пин open3d==0.11.2)
uv pip install --no-deps mesh_processing_tools==1.0.4

# Все остальные зависимости
uv pip install -r requirements-local.txt

# Сам neurd в dev-режиме
uv pip install -e .

echo "Done. Run: source .venv/bin/activate && pytest tests/unit/"
```

---

### Шаг 4 — Новый Dockerfile на python:3.12-slim (2-3 ч)

Переписать `docker/Dockerfile`:

```dockerfile
FROM python:3.12-slim

# Системные зависимости open3d (headless) + git для git-зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# mesh_processing_tools: чистый Python, --no-deps обходит open3d==0.11.2
RUN uv pip install --system --no-deps mesh_processing_tools==1.0.4

# Все зависимости mesh_tools с современными версиями
RUN uv pip install --system \
    open3d>=0.19 \
    pykdtree>=1.4 \
    pymeshfix>=0.16 \
    h5py \
    tqdm \
    meshparty>=2.0

# Авторские пакеты NEURD
RUN uv pip install --system \
    datasci_stdlib_tools==1.0.1 \
    machine_learning_tools==1.0.0 \
    neuron_morphology_tools==1.0.1 \
    graph_nx_tools==1.0.0 \
    code_structure_tools==1.0.2

# Опциональные зависимости для motif-анализа
RUN uv pip install --system \
    git+https://github.com/reimerlab/tamarind.git \
    git+https://github.com/reimerlab/dotmotif/

RUN uv pip install --system pytest pytest-mock
```

**Что убрали:**
- `FROM celiib/mesh_tools:v4` — больше не нужен
- `datajoint` — убран из base (есть в `extras_require[connectome]`)
- `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL` — не нужен на 3.12
- Жёсткий пин `trimesh==3.22.3` — снимаем, ставим `>=4`

---

### Шаг 5 — Удалить Docker полностью (после Шагов 1-4) (1 ч)

После того как локальный запуск работает:

1. Обновить [docker/README_TESTS.md](docker/README_TESTS.md) — добавить секцию
   «Локальный запуск без Docker».
2. В [docker/docker-compose.yml](docker/docker-compose.yml) — заменить
   `image: celiib/mesh_tools:v4` на `build: .` в сервисе `test`.
3. Добавить в README.md корневой секцию быстрого старта:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   bash scripts/install_local.sh
   pytest tests/unit/
   ```

---

## Какие тесты написать

### Тест 1 — Smoke-импорт (обязателен перед всем остальным)

```python
# tests/unit/test_env_compat.py
"""Verify that the package imports cleanly on modern Python / numpy 2."""

def test_neurd_imports_without_error():
    import neurd  # triggers shims

def test_datasci_tools_imports_after_shim():
    import neurd  # shim должен сработать первым
    from datasci_tools import numpy_dep as np
    assert hasattr(np, 'float_')

def test_mesh_tools_skeleton_utils_imports_without_ipyvolume(monkeypatch):
    """skeleton_utils должен импортироваться даже без ipyvolume."""
    import sys
    # убедимся что ipyvolume НЕ установлен (или заглушён)
    monkeypatch.delitem(sys.modules, 'ipyvolume', raising=False)
    import neurd  # shim вставит stub
    from mesh_tools import skeleton_utils  # не должно падать

def test_mesh_tools_trimesh_utils_imports():
    from mesh_tools import trimesh_utils  # нужен open3d>=0.19
```

### Тест 2 — Совместимость API open3d 0.19

`trimesh_utils.py` из `mesh_tools` использует open3d. Нужно проверить
что функции которые мы зовём из neurd не ломаются:

```python
# tests/unit/test_mesh_tools_compat.py
"""Check that mesh_tools submodules work with modern dependency versions."""
import pytest

def test_trimesh_utils_imports():
    from mesh_tools import trimesh_utils as tu
    assert hasattr(tu, 'mesh_from_vertices_faces')  # основная функция

def test_skeleton_utils_imports():
    from mesh_tools import skeleton_utils as sk
    assert hasattr(sk, 'calculate_skeleton_distance')

def test_meshparty_version():
    import meshparty
    # Убедиться что >= 2.0 стоит
    major = int(meshparty.__version__.split('.')[0])
    assert major >= 2
```

### Тест 3 — numpy 2 совместимость

```python
# tests/unit/test_numpy_compat.py
import numpy as np

def test_numpy_version_is_2():
    major = int(np.__version__.split('.')[0])
    assert major >= 2, f"Expected numpy>=2, got {np.__version__}"

def test_numpy_float64_operations():
    """Базовые операции которые использует datasci_tools."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert arr.mean() == 2.0
    assert np.float64(1.5) == 1.5

def test_no_numpy_float_deprecation_warning():
    """np.float_ должен работать после shima."""
    import neurd  # активирует shim
    assert np.float_ == np.float64
```

### Тест 4 — Пайплайн без тяжёлых deps (интеграционный, после Шагов 1-3)

```python
# tests/integration/test_minimal_import.py
"""
Verify that the mesh segmentation core can be imported without
datajoint / seaborn / ipyvolume / torch being installed.
"""
import sys
import pytest

@pytest.fixture(autouse=True)
def no_heavy_deps(monkeypatch):
    """Remove heavy optional packages from sys.modules."""
    for pkg in ('datajoint', 'seaborn', 'ipyvolume', 'torch', 'torch_geometric'):
        monkeypatch.delitem(sys.modules, pkg, raising=False)

def test_neuron_utils_importable():
    from neurd import neuron_utils

def test_preprocess_neuron_importable():
    from neurd import preprocess_neuron

def test_volume_utils_importable():
    from neurd import volume_utils
```

---

## Метрики успеха

| Метрика | До Фазы 3 | После Шага 1-2 | После Шага 3-4 |
|---------|-----------|----------------|----------------|
| `python3 -c "import neurd"` без Docker | ❌ падает | ✅ | ✅ |
| `pytest tests/unit/` без Docker | ❌ | ✅ | ✅ |
| Зависимость от `celiib/mesh_tools:v4` | ✅ (есть) | ✅ (есть) | ❌ (убрана) |
| Python версия в Docker | 3.8 | — | **3.12** |
| Время сборки Docker-образа | ~10 мин (open3d==0.11.2 тяжёлый) | — | ~3-5 мин |
| numpy версия | <2 (ограничено) | **≥2** | **≥2** |

---

## Порядок реализации (рекомендуемый)

```
Шаг 1: numpy shim в __init__.py → проверить локально
Шаг 2: ipyvolume stub в __init__.py → проверить локально  
Тест: pytest tests/unit/ — должен быть зелёным локально
Шаг 3: install_local.sh + requirements-local.txt
Шаг 4: новый Dockerfile → docker compose run --rm test
Шаг 5: удалить FROM celiib/mesh_tools:v4 навсегда
```

После Шага 2 + локального запуска тестов — Docker становится опциональным,
не обязательным инструментом (CI/CD или изолированная воспроизводимая сборка).

---

## Риски

| Риск | Вероятность | Mitigation |
|------|-------------|------------|
| `mesh_tools/trimesh_utils` использует open3d API которое изменилось в 0.19 | средняя | написать test_mesh_tools_compat.py до смены образа |
| `meshparty 2.0` сломал API относительно 1.x | низкая | проверить `from meshparty import trimesh_io` — используется в `neurd/neuron.py` |
| `datasci_tools` использует что-то ещё удалённое в numpy 2, кроме `float_` | низкая | shim покрывает `float_/int_/complex_/bool_` — самые частые кандидаты |
| open3d 0.19 требует другие системные либы | низкая | `libgl1 libglib2.0-0 libgomp1` — стандартный набор |
