# Dependencies for EVO2 model

```bash
mm create -n pbi-evo2
mm activate pbi-evo2
mm install "python==3.12.*"
```

```bash
mm install -c nvidia cuda-nvcc cuda-cudart-dev
mm install -c conda-forge transformer-engine-torch=2.3.0
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
pip install ipykernel ipywidgets
```