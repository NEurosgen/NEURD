# the original docker build is the docker build from the mesh_tools repo:
# https://github.com/bacelii/mesh_tools/tree/main/docker



FROM celiib/mesh_tools:v4

RUN pip3 install uv

# Разрешаем установку устаревшего пакета sklearn
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

RUN uv pip install --system \
    datasci_stdlib_tools==1.0.1 \
    machine_learning_tools==1.0.0 \
    neuron_morphology_tools==1.0.1 \
    graph_nx_tools==1.0.0 \
    mesh_processing_tools==1.0.4 \
    code_structure_tools==1.0.2 


RUN uv pip install --system \
    git+https://github.com/reimerlab/tamarind.git \
    git+https://github.com/reimerlab/dotmotif/

RUN uv pip install --system pytest pytest-mock
