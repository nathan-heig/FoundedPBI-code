# pbi

## Dependencies installation

### CUDA 12.4
The EVO model has a dependency on `flash-attn`, which requires CUDA>=11.7.

To install CUDA==12.4, run (replace `mm` with conda, if you use that):
```bash
mm install cuda-libraries-dev cuda-nvcc cuda-nvtx cuda-cupti nccl -c nvidia/label/cuda-12.4.0
```

## DNABERT2
The DNABERT2 model needs the last version of `triton`, so you need to install it after the requirements.txt.

As per my testing, it seems to work fine even if pytorch complains.

## EVO
The EVO2 model can only be run on H100 and H200 GPUs, so we cannot use it. See [here](https://docs.nvidia.com/nim/bionemo/evo2/latest/prerequisites.html) for the hardware requirements.