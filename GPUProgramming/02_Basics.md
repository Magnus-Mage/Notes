# Basics of CUDA

- CUDA is the Compute unified device architecture (CUDA) can be thought of as a programming language for nvidia gpus.
- CUDA libs => cuDNN, cuBLAS, cutlass, cuFFT for fast convolutions.
- Writing the kernel yourself based on the hardware architecutre. (Nvidia still does it anyways, but better to write your own optimisations for better performance)

## Compiler

CUDA uses `NVCC` compiler from the CUDA toolkit to compile and run the CUDA source code.

Steps on how to install CUDA toolkit => [Installation Guide](01_Installation_Guide.md)

Confirm NVCC is installed on your environment :
```bash
nvcc --version 

# or nvidia-smi
```
Output should show the version of the  CUDA compiler or the nvidia resources.

![CUDA Compilation Trajectory](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/_images/cuda-compilation-from-cu-to-executable.png)


