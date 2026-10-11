from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent

def get_install_requires(filepath=None):
    """Read runtime requirements relative to the setup file, without comments."""
    folder = ROOT if filepath is None else Path(filepath).resolve().parent
    return [line.strip() for line in (folder / 'requirements.txt').read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]

# read in version number into __version__
with (ROOT / 'neurd/version.py').open() as f:
    exec(f.read())

setup(
    name='neurd',  # import name stays `neurd`: this is a fork, not a rename
    version=__version__,
    description=(
        'Slim single-soma fork of NEURD: mesh -> somas/limbs/branches + skeletons '
        '+ raw spines, optimized for wall time and peak RAM'
    ),
    long_description=(ROOT / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    project_urls={
        'Source': "https://github.com/NEurosgen/NEURD/",
        'Upstream': "https://github.com/reimerlab/NEURD/",
    },
    author='Eugen Didenko',
    # Upstream NEURD by Brendan Celii <brendanacelii@gmail.com> (reimerlab/NEURD).
    # 3.13+ has no open3d wheel; 3.9 lacks numpy-2 wheels for parts of the stack.
    python_requires='>=3.10,<3.13',
    packages=find_packages(include=['neurd', 'neurd.*']),
    include_package_data=True,
    install_requires=get_install_requires(),
    # pip install -e ".[dev]"      — test runner for the tests/unit gate
    # pip install -e ".[poisson]"  — MeshLab-accurate Poisson (NEURD_REAL_POISSON=1)
    # pip install -e ".[viz]"      — seaborn + ipyvolume interactive 3D
    extras_require={
        'dev': [
            'pytest',
            'pytest-mock',
        ],
        # Only needed with NEURD_REAL_POISSON=1; the default path no-ops Poisson
        # (see neurd/__init__.py) so pymeshlab is not a runtime requirement.
        'poisson': [
            'pymeshlab',
        ],
        'viz': [
            'seaborn>=0.12.2',
            'ipyvolume>=0.6.3',
        ],
        'all': [
            'pytest',
            'pytest-mock',
            'pymeshlab',
            'seaborn>=0.12.2',
            'ipyvolume>=0.6.3',
        ],
    },
)

