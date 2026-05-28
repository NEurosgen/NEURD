# NEURD

---

NEURD: A mesh decomposition framework for automated proofreading and morphological analysis of neuronal EM reconstructions

publication: https://www.nature.com/articles/s41586-025-08660-5

## Setup: Local install (Python 3.10–3.12)

This fork runs NEURD directly in a virtualenv or conda env — no Docker required.

### Quick start

```bash
# 1. Create env (any of: venv, conda, pyenv) on Python 3.10–3.12
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install everything
bash scripts/install_local.sh

# 3. Verify
pytest tests/unit/
```

The install script handles the one fiddly bit: `mesh_processing_tools` pins
`open3d==0.11.2` in its metadata, but we use modern `open3d>=0.19`, so the
script installs it with `--no-deps` and pulls dependencies from
[requirements-local.txt](requirements-local.txt).

### Notes

- Python **3.13+** is not yet supported (no `open3d` wheels on PyPI).
- On a fresh OS you may need: `apt install libgl1 libglib2.0-0 libgomp1`
  (open3d runtime libs).
- The mesh-processing pipeline (decimation, soma extraction, decomposition) shells
  out to `xvfb-run meshlabserver`, so running it needs MeshLab + `xvfb`
  (`apt install meshlab xvfb`). The import/unit tests do **not** require these;
  the integration test (`tests/integration/`) skips mesh stages gracefully if absent.
- Optional extras for cloud / visualization workflows are declared in
  `setup.py` under `extras_require` (`[connectome]`, `[viz]`).

## Documentation

Documentation Site: https://reimerlab.github.io/NEURD/

### Tutorials

All of the tutorials made for showing the decomposition/autoproofreading pipeline (and other features like spine detection and proximity detection) are in .ipynb files inside Applications>Tutorials.

#### Highlighted Tutorials:

---

1. Auto Proofreading Pipeline:

   - Multi Soma: Applications/Tutorials/Auto_Proof_Pipeline/Double_Soma/neuron_pipeline_vp5_double_soma.ipynb
   - Single Soma Excitatory: Applications/Tutorials/Auto_Proof_Pipeline/Single_Soma_Exc/neuron_pipeline_vp5_single_demo_exc.ipynb
   - Single Soma Inhibitory: Applications/Tutorials/Auto_Proof_Pipeline/Single_Soma_Inh/neuron_pipeline_vp5_single_demo_inh.ipynb

2. Neuron Object:

   - Hierarchical Organization and Access: Applications/Tutorials/Neuron_Features/Neuron_Limb_Branch_Hierarchical_Data_Structure.ipynb
   - Neuron Feature Tutorial: Applications/Tutorials/Neuron_Features/Neuron_Features_Tutorial.ipynb
     \*\*\* See Neuron_Feature_Documentation sheet below for detailed descriptions \*\*\*

3. SWC Output and Analysis:

   - SWC Output and Analysis with 3rd Party Software: Applications/Tutorials/SWC_Output_and_Analysis/SWC_output_and_morphopy_analysis.ipynb

4. Volume Data Interface (VDI) Override Implementations:

   - H01 (Human Dataset) VDI Override: Applications/Tutorials/VDI_override/Tutorial_Making_Vdi_Override_H01.ipynb
   - Fake Data VDI Override: Applications/Tutorials/VDI_override/Tutorial_Making_Vdi_Override_Whale.ipynb

5. Visualizations:

   - Skeleton and Compartments of Auto Proofread Neuron: Applications/Tutorials/Visualizing_Auto_Proof_Neurons/Visualizing_Neuron_Skeletons_and_Compartments.ipynb

#### Documentation sheets:

---

1. Neuron_Feature_Documentation:
   https://docs.google.com/spreadsheets/d/1DBFTMUY7RpRoDQM3TWEb4aZoJGc-7cxq0_D-zkJ4JSE/edit?usp=sharing
2. NEURD module overview:
   https://docs.google.com/spreadsheets/d/1B5FqA1jQjadnEuQPjmbhHZFthm21NW3tGNcoLVrEUW4/edit?usp=sharing
3. NEURD paper N Table (for figures in publication)
   https://docs.google.com/spreadsheets/d/1OHeZjenEdYGDCl_5wM6wTdxFV_ouT3sQoJw5lVvgatg/edit?usp=sharing
4. NEURD submodule parameter documentation:
   https://docs.google.com/spreadsheets/d/ 1hrhCo4NKqTowep_ju-mICGHFp33TfWbS96tHEfbtZEs/edit?usp=sharing
