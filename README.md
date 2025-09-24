# pbi

## Dependencies installation

### CUDA 12.4
The EVO model has a dependency on `flash-attn`, which requires CUDA>=11.7.

To install CUDA==12.4, run (replace `mm` with conda, if you use that):
```bash
mm install cuda-libraries-dev cuda-nvcc cuda-nvtx cuda-cupti nccl -c nvidia/label/cuda-12.4.0
```