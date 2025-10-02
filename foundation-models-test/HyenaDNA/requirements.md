```bash
mm create -n pbi-hyenadna
mm activate pbi-hyenadna
mm install "python==3.10.18"
mm install cuda-libraries-dev cuda-nvcc cuda-nvtx cuda-cupti nccl -c nvidia/label/cuda-12.4.0
```

```bash
pip install -r requirements-hyenadna.txt
```

```bash
git clone --recurse-submodules https://github.com/HazyResearch/hyena-dna.git && cd hyena-dna
```

```bash
pip install -r requirements.txt
```

```bash
git submodule update --init
cd flash-attention
git submodule update --init
pip install -e . --no-build-isolation
cd csrc/layer_norm && pip install . --no-build-isolation
```