from os import path
from pathlib import Path
from setuptools import setup, find_packages
from typing import List

here = path.abspath(path.dirname(__file__))

def get_install_requires(filepath=None):
    """Returns requirements.txt parsed to a list, minus comments and blank lines."""
    if filepath is None:
        filepath = "./"
    fname = Path(filepath).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            for line in f.read().splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    targets.append(line)

    targets += get_links()
    return targets

def get_links():
    return [
        #"datasci_tools @ git+https://github.com/bacelii/datasci_tools.git'"
    ]

def get_long_description(filepath='README.md'):
    try:
        import pypandoc
        long_description = pypandoc.convert_file(filepath, 'rst') 
    except:
        print("\n\n\n****Need to install pypandoc (and if havent done so install apt-get install pandoc) to make long description clean****\n\n\n")
        
        long_description = Path("README.md").read_text()
        
    return long_description

# read in version number into __version__
with open(path.join(here, 'neurd', 'version.py')) as f:
    exec(f.read())

setup(
    name='neurd',  # import name stays `neurd`: this is a fork, not a rename
    version=__version__,
    description=(
        'Slim single-soma fork of NEURD: mesh -> somas/limbs/branches + skeletons '
        '+ raw spines, optimized for wall time and peak RAM'
    ),
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    project_urls={
        'Source': "https://github.com/NEurosgen/NEURD/",
        'Upstream': "https://github.com/reimerlab/NEURD/",
    },
    author='Eugen Didenko',
    # Upstream NEURD by Brendan Celii <brendanacelii@gmail.com> (reimerlab/NEURD).
    # 3.13+ has no open3d wheel; 3.9 lacks numpy-2 wheels for parts of the stack.
    python_requires='>=3.10,<3.13',
    packages=find_packages(),  #teslls what packages to be included for the install
    # package_data = {
    #     'neurd':['neurd/model_data/*'],
    # },
    include_package_data=True,
    install_requires=get_install_requires(), #external packages as dependencies
    # dependency_links = get_links(),
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
    
    # if have a python script that wants to be run from the command line
    entry_points={
        #'console_scripts': ['pipeline_download=Applications.Eleox_Data_Fetch.Eleox_Data_Fetcher_vp1:main']
    },
    scripts=[], 
    
)

