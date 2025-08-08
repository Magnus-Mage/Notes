
# NVIDIA ARCHITECTURE HISTORY


NVIDIA's GPU architectures have evolved significantly over the years, each introducing new features, performance enhancements, and compatibility considerations for libraries and frameworks. Below is a comprehensive overview of these architectures, their release timelines, key differences, and compatibility with major libraries and frameworks.

---

##  NVIDIA GPU Architecture Generations

### **1. Celsius (1999)**
- Release Date: 1999
- Graphic Features:
  - DirectX 7.0
  - [OpenGL](https://en.wikipedia.org/wiki/OpenGL "OpenGL") 1.2 
  - Max VRAM size bumped to 128MB
- First ever "GeForce" product line release with GeForce 256.
- Improves upon its predecessor (RIVA TNT2) by increasing the number of fixed pipelines, offloading host geometry calculations to a [hardware transform and lighting](https://www.bing.com/ck/a?!&&p=ccc2e5a7c5c83a88ebc65298f4833d7f42ad52f5cca20db23527673dedcfccccJmltdHM9MTc0NzI2NzIwMA&ptn=3&ver=2&hsh=4&fclid=1819c88d-b1d8-6e3e-1539-dd20b02a6fa0&u=a1L3NlYXJjaD9xPVRyYW5zZm9ybSUyMGFuZCUyMGxpZ2h0aW5nJTIwd2lraXBlZGlhJmZvcm09V0lLSVJF&ntb=1) (T&L) engine, and adding hardware [motion compensation](https://www.bing.com/ck/a?!&&p=fbc117aca51082b5f5f7fbe05d528e0606d08b4ef9e179eebc3d354730344b56JmltdHM9MTc0NzI2NzIwMA&ptn=3&ver=2&hsh=4&fclid=1819c88d-b1d8-6e3e-1539-dd20b02a6fa0&u=a1L3NlYXJjaD9xPU1vdGlvbiUyMGNvbXBlbnNhdGlvbiUyMHdpa2lwZWRpYSZmb3JtPVdJS0lSRQ&ntb=1) for [MPEG-2](https://www.bing.com/ck/a?!&&p=8f7a39902a0cee3e4b7c11769df14a274d1961186e8a63120b211af050f7b0deJmltdHM9MTc0NzI2NzIwMA&ptn=3&ver=2&hsh=4&fclid=1819c88d-b1d8-6e3e-1539-dd20b02a6fa0&u=a1L3NlYXJjaD9xPU1QRUctMiUyMHdpa2lwZWRpYSZmb3JtPVdJS0lSRQ&ntb=1) video.
- It offered a notable leap in 3D PC gaming performance.
- It was the first fully [Direct3D 7](https://en.wikipedia.org/wiki/Microsoft_Direct3D "Microsoft Direct3D")-compliant 3D accelerator.

### **2. Rankine (2003)**
- Release Date: 2003
- Graphic Features:
	- DirectX 9.0a
	- [OpenGL](https://en.wikipedia.org/wiki/OpenGL "OpenGL") 
	- Shader Model 2.0a
	- Vertex Shader 2.0a
	- Max VRAM size bumped to 256MB
- GeForce 5(5xxxx) series was launched first with this architecture.
- The GeForce FX 5100 was a graphics card by NVIDIA built on the 150 nm process, and based on the NV34 graphics processor, the card supports DirectX 9.0a.
- It is compliant with Shader Model 2.0/2.0A, allowing more flexibility in complex shader/fragment programs and much higher arithmetic precision.
-  It supports a number of new memory technologies, including [DDR2](https://en.wikipedia.org/wiki/DDR2_SDRAM "DDR2 SDRAM"), [GDDR2](https://en.wikipedia.org/wiki/GDDR-2 "GDDR-2") and [GDDR3](https://en.wikipedia.org/wiki/GDDR-3 "GDDR-3") and saw Nvidia's first implementation of a memory data bus wider than 128 bits.
### **3. Curie Architecture (2004)**
- Release Date: 2004
- Graphic Features:
	- DirectX 9.0c (9_3)
	- [OpenGL](https://en.wikipedia.org/wiki/OpenGL "OpenGL") 2.1
	- Shader Model 3.0
	- [Nvidia PureVideo](https://en.wikipedia.org/wiki/Nvidia_PureVideo "Nvidia PureVideo") (first generation)
	- Reintroduced support for [Z compression](https://en.wikipedia.org/wiki/Compress#Special_output_format "Compress")
	- Hardware support for [MSAA](https://en.wikipedia.org/wiki/Multisample_anti-aliasing "Multisample anti-aliasing") [anti-aliasing](https://en.wikipedia.org/wiki/Anti-aliasing_filter "Anti-aliasing filter") algorithm (up to 4x)
- Shortly after the release of the GeForce FX series came the 6 series (aka NV40). GeForce 6 was the start of Nvidia pushing SLI technology allowing people to combine more than one graphics card for more power.
- The flagship of this range was the GeForce 6800 Ultra, a graphics card which boasted 222 million transistors, 16-pixel superscalar pipelines and six vertex shaders. It had Shader Model 3.0 support and was compliant with both Microsoft DirectX 9.0c and OpenGL 2.0.
- The series also featured Nvidia PureVideo technology and was able to decode H.264, VC-1, WMV and MPEG-2 videos with reduced CPU use.
- One of the most highly successful GPU series of the time.
### 4. **Tesla (2006)**

* **Release Date:** 2006
* Graphic Features:
	* Direct3D 10
	* Shader Model 4.0
	* OpenGL 2.1 (Later supports have OpenGL 3.3)
* **Key Features:** Introduced CUDA, enabling general-purpose GPU computing.
* **Compatibility:** Laid the foundation for CUDA-based applications.
* **Tesla** is the codename for a GPU [microarchitecture](https://en.wikipedia.org/wiki/Microarchitecture "Microarchitecture") developed by [Nvidia](https://en.wikipedia.org/wiki/Nvidia "Nvidia"), and released in 2006, as the successor to [Curie](https://en.wikipedia.org/wiki/Curie_\(microarchitecture\) "Curie (microarchitecture)") microarchitecture.
* As Nvidia's first microarchitecture to implement unified shaders, it was used with [GeForce 8 series](https://en.wikipedia.org/wiki/GeForce_8_series "GeForce 8 series"), [GeForce 9 series](https://en.wikipedia.org/wiki/GeForce_9_series "GeForce 9 series"), [GeForce 100 series](https://en.wikipedia.org/wiki/GeForce_100_series "GeForce 100 series"), [GeForce 200 series](https://en.wikipedia.org/wiki/GeForce_200_series "GeForce 200 series"), and [GeForce 300 series](https://en.wikipedia.org/wiki/GeForce_300_series "GeForce 300 series") of GPUs, collectively manufactured in [90 nm](https://en.wikipedia.org/wiki/90_nm "90 nm"), [80 nm](https://en.wikipedia.org/w/index.php?title=80_nm&action=edit&redlink=1 "80 nm (page does not exist)"), [65 nm](https://en.wikipedia.org/wiki/65_nm "65 nm"), [55 nm](https://en.wikipedia.org/w/index.php?title=55_nm&action=edit&redlink=1 "55 nm (page does not exist)"), and [40 nm](https://en.wikipedia.org/w/index.php?title=40_nm&action=edit&redlink=1 "40 nm (page does not exist)").
* Introduced Nvidia NVDEC and NVENC for video decompression and compression.
* Introduced better anti aliasing algorithms and trilinear texture filtering.
* Also, Introduced C support to allow developers use GPU without having to learn a new language.
### 5. **Fermi (2010)**

* **Release Date:** 2010
* Graphic Features:
	* Improve Double Precision Performance
	* ECC support for data center support
	* True Cache Hierarchy for shared memory
	* More Shared Memory—many CUDA programmers requested more than 16 KB of SM shared memory to speed up their applications.
	* Faster Context Switching for faster context switches between application programs, faster graphics and compute interpolation.
	* Faster Atomic Operations for read write and modify atomic operations.
	* Third Generation Streaming Multiprocessor (SM), Second Generation Parallel Thread Execution ISA, Improved Memory Subsystem, NVIDIA GigaThreadTM Engine
- Fermi was followed by [Kepler](https://en.wikipedia.org/wiki/Kepler_\(microarchitecture\) "Kepler (microarchitecture)"), and used alongside Kepler in the [GeForce 600 series](https://en.wikipedia.org/wiki/GeForce_600_series "GeForce 600 series"), [GeForce 700 series](https://en.wikipedia.org/wiki/GeForce_700_series "GeForce 700 series"), and [GeForce 800 series](https://en.wikipedia.org/wiki/GeForce_800_series "GeForce 800 series"), in the latter two only in [mobile](https://en.wikipedia.org/wiki/Mobile_computer "Mobile computer") GPUs.
-  All desktop Fermi GPUs were manufactured in 40nm, mobile Fermi GPUs in 40nm and [28nm](https://en.wikipedia.org/wiki/28nm "28nm").
- Each SM features 32 single-precision CUDA cores, 16 load/store units, four Special Function Units (SFUs), a 64 KB block of high speed on-chip memory (see L1+Shared Memory subsection) and an interface to the L2 cache (see L2 Cache subsection).
- Introduced Load/Store units and SFUs.
### 6. **Kepler (2012)**

* **Release Date:** 2012
* **Key Features:** 
	*  [PCI Express 3.0](https://en.wikipedia.org/wiki/PCI_Express#PCI_Express_3.0 "PCI Express") interface
	- [DisplayPort](https://en.wikipedia.org/wiki/DisplayPort "DisplayPort") 1.2
	- [HDMI](https://en.wikipedia.org/wiki/HDMI "HDMI") 1.4a 4K x 2K video output
	- [PureVideo VP5](https://en.wikipedia.org/wiki/Purevideo "Purevideo") hardware video acceleration (up to 4K x 2K H.264 decode)
	- Hardware [H.265](https://en.wikipedia.org/wiki/H.265 "H.265") decoding
	- Hardware [H.264](https://en.wikipedia.org/wiki/H.264 "H.264") encoding acceleration block (NVENC)
	- Support for up to 4 independent 2D displays, or 3 stereoscopic/3D displays (NV Surround)
	- Next Generation Streaming Multiprocessor (SMX)
	- Polymorph-Engine 2.0
	- Simplified Instruction Scheduler
	- Bindless Textures
	- [CUDA](https://en.wikipedia.org/wiki/CUDA "CUDA") Compute Capability 3.0 to 3.5
	- GPU Boost (Upgraded to 2.0 on GK110)
	- TXAA Support
	- Manufactured by [TSMC](https://en.wikipedia.org/wiki/TSMC "TSMC") on a 28 nm process
	- New Shuffle Instructions
	- Dynamic Parallelism
	- Hyper-Q (Hyper-Q's MPI functionality reserve for Tesla only)
	- Grid Management Unit
	- Nvidia GPUDirect (GPU Direct's RDMA functionality reserve for Tesla only)
	* **Compatibility:** Supported newer versions of CUDA, enhancing performance in parallel computing tasks.
- The theoretical single-precision processing power of a Kepler GPU in [GFLOPS](https://en.wikipedia.org/wiki/GFLOPS "GFLOPS") is computed as 2 (operations per FMA instruction per CUDA core per cycle) × number of CUDA cores × core clock speed (in GHz).
- Note that like the previous generation [Fermi](https://en.wikipedia.org/wiki/Fermi_\(microarchitecture\)#Performance "Fermi (microarchitecture)"), Kepler is not able to benefit from increased processing power by dual-issuing MAD+MUL like [Tesla](https://en.wikipedia.org/wiki/Tesla_\(microarchitecture\)#Performance "Tesla (microarchitecture)") was capable of.
- The GeForce 600 series introduced Nvidia's Kepler architecture which was designed to increase performance per watt while also improving upon the performance of the previous Fermi microarchitecture.

### 7. **Maxwell (2014)**

* **Release Date:** 2014
* Released with 900 Series – GTX 960, GTX 970, GTX 980
* Graphic Features:
	* Support for CUDA 5.0
	* Increased L2 cache memory size to 2MiB and decreased memory bandwidth to 128 bit reducing die area, cost, and power draw.
	* Dynamic Super Resolution, Third Generation Delta Color Compression,Multi-Pixel Programming Sampling,) Nvidia VXGI (Real-Time-Voxel-[Global Illumination](https://en.wikipedia.org/wiki/Global_illumination "Global illumination")), VR Direct, Multi-Projection Acceleration, Multi-Frame Sampled Anti-Aliasing(MFAA) (however, support for Coverage-Sampling Anti-Aliasing(CSAA) was removed),and Direct3D12 API at Feature Level 12_1. HDMI 2.0 support was also added.
	*  Dynamic Parallelism and HyperQ, two features in GK110/GK208 GPUs, are also supported across the entire Maxwell product line.
	* Nvidia's video encoder, NVENC, was upgraded to be 1.5 to 2 times faster than on Kepler-based GPUs, meaning it can encode video at six to eight times playback speed.
* **Compatibility:** Optimized for gaming and graphics applications; backward compatibility with CUDA 5.0.([Wikipedia][1])
* The theoretical single-precision processing power of a Maxwell GPU in [FLOPS](https://en.wikipedia.org/wiki/FLOPS "FLOPS") is computed as 2 (operations per FMA instruction per CUDA core per cycle) × number of CUDA cores × core clock speed (in Hz).
* Maxwell was a big jump for the company as the GTX 980 came with 5200 million transistors and 4 GB of GDDR5 memory.
* Maxwell was also based on the TSMCs 28 Nm manufacturing process.
* The graphics card came with an 1127 Mhz clock speed, 7 Gbps memory throughput, and incredible gaming performance. With a TDP of just 165 Watts, it offered unmatched performance that was never seen before.


### 8. **Pascal (2016)**

* **Release Date:** 2016
* **Key Features:** First architecture to support CUDA Compute Capability 6.0.
* **Compatibility:** Enhanced performance for AI and deep learning frameworks; supported CUDA 6.0.
* Graphic Features:
	* In Pascal, a SM (streaming multiprocessor) consists of between 64-128 CUDA cores, depending on if it is GP100 or GP104. [Maxwell](https://en.wikipedia.org/wiki/Maxwell_\(microarchitecture\) "Maxwell (microarchitecture)") contained 128 CUDA cores per SM; Kepler had 192, Fermi 32 and Tesla 8.
	* The GP100 SM is partitioned into two processing blocks, each having 32 single-precision CUDA cores, an instruction buffer, a warp scheduler, 2 texture mapping units and 2 dispatch units.
	* - Unified memory — a memory architecture where the CPU and GPU can access both main system memory and memory on the graphics card with the help of a technology called "Page Migration Engine".
	* [NVLink](https://en.wikipedia.org/wiki/NVLink "NVLink") — a high-bandwidth bus between the CPU and GPU, and between multiple GPUs. Allows much higher transfer speeds than those achievable by using PCI Express; estimated to provide between 80 and 200 GB/s
	* 16-bit ([FP16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format "Half-precision floating-point format")) floating-point operations (colloquially "half precision") can be executed at twice the rate of 32-bit floating-point operations ("single precision")
	* - More registers — twice the amount of registers per CUDA core compared to Maxwell.
	* DirectX and Direct3D 12.0 support.
	* OpenGL 4.6 support.
	* Vulkan 1.3 support.
* The graphics card was manufactured using the 16 Nm manufacturing process from TSMC.
* The GPU came with 7200 million transistors, 8 GB of GDDR5X VRAM, and 10 Gbps memory throughput. It was one of the most impressive graphics cards and pushed the gaming industry to embrace a 4K resolution.
* The card had a boost clock speed of 1733 Mhz and a TDP of just 180 Watts and delivered much more power than the previous generations.

### 9. **Volta (2017)**

* **Release Date:** 2017
* Graphic Features:
	* Tensor Cores were indtroduced.
	* TSMC's 12 nm FinFET process, allowing 21.1 billion transistors.
	* High BandWidth Memory.
	* PureVideo Feature Set I hardware video decoding
	* Volta GV100 Full GPU with 84 SM units
	* FP16 AND FP32 input and compute processes.
* **Key Features:** Introduced Tensor Cores for deep learning acceleration.
* **Compatibility:** Supported CUDA Compute Capability 7.0; optimized for AI workloads.([Wikipedia][1])
* After two USPTO proceedings on July 3, 2023 Nvidia lost the Volta trademark application in the field of artificial intelligence. The Volta trademark owner remains Volta Robots, a company specialized in AI and vision algorithms for robots and unmanned vehicles.

### 10. **Turing (2018)**

* **Release Date:** 2018
* Graphic Features:
	* DirectX 12 and Direct3D 12
	* Shader Model 6.7
	* OPenGL 4.6 support
	* CUDA compute capability 7.5
	* Vulkan 1.3
	* Tensor(AI) cores.
	* DLSS (Deep Learning Super Sampling)
	* Memory Control with GDDR6/HBM2
	* NVENC hardware Encoding
* **Key Features:** Introduced real-time ray tracing and Tensor Cores.
* **Compatibility:** Supported CUDA Compute Capability 7.5; enhanced support for AI and graphics applications.([The Verge][2])
* The ray-tracing performed by the RT cores can be used to produce reflections, refractions and shadows, replacing traditional raster techniques such as [cube maps](https://en.wikipedia.org/wiki/Cube_mapping "Cube mapping") and [depth maps](https://en.wikipedia.org/wiki/Depth_map "Depth map"). 
* Instead of replacing rasterization entirely, however, the information gathered from ray-tracing can be used to augment the shading with information that is much more [photo-realistic](https://en.wikipedia.org/wiki/Photorealism "Photorealism"), especially in regards to off-camera action. 
* Nvidia said the ray-tracing performance increased about 8 times over the previous consumer architecture, Pascal.
* GeForce MX series, GeForce 16 series, GeForce 20 series, Nvidia Quadro etc used this architecture.
* Notably, these graphics cards were the first-gen of RTX cards and saw Nvidia pushing [ray tracing](https://www.pocket-lint.com/games/news/nvidia/148279-what-is-ray-tracing-and-what-hardware-and-games-support-it/) as the main selling point.

### 11. **Ampere (2020)**

* **Release Date:** 2020
* Graphic Features:
	* Directx and Direct3D 12
	* Shader Model 6.8
	* OpenGL 4.6
	* CUDA compute capability 8.6
	* Vulkan 1.3
	* NVENC hardware encoding
	* memory support with GDDR6 and GDDR6X
	* Hugh Bandwidth Memory 2 on A100 40 GB
* **Key Features:** Enhanced Tensor Cores and support for CUDA Compute Capability 8.0.
* **Compatibility:** Optimized for AI, gaming, and data center applications; supported CUDA 8.0.
* Nvidia announced the Ampere architecture [GeForce 30 series](https://en.wikipedia.org/wiki/GeForce_30_series "GeForce 30 series") consumer GPUs at a GeForce Special Event on September 1, 2020.
* Mobile RTX graphics cards and the RTX 3060 based on the Ampere architecture were revealed on January 12, 2021.
* - [PureVideo](https://en.wikipedia.org/wiki/Nvidia_PureVideo "Nvidia PureVideo") feature set K hardware video decoding with [AV1](https://en.wikipedia.org/wiki/AV1 "AV1") hardware decoding for the GeForce 30 series and feature set J for A100
* GeForce MX series(mobile), GeForce 20 series(mobile), GeForce 30 series, Nvidia Workstation GPUs, Nvidia Data Center GPUs, Tegra SoCs used this design.

### 12. **Ada Lovelace (2022)**

- **Release Date:** October 12, 2022
- **Graphic Features:**
    - DirectX 12 Ultimate (Feature Level 12_2)
    - Shader Model 6.8
    - OpenCL 3.0
    - OpenGL 4.6
    - CUDA Compute Capability 8.9
    - Vulkan 1.3
    - NVENC hardware encoding (H.264, H.265, AV1)
    - Memory support with GDDR6 and GDDR6X
    - PCIe 4.0

- **Key Features:** Introduction of DLSS 3 with AI-generated frames, enhanced ray tracing capabilities, and improved power efficiency.
- **Compatibility:** Optimized for gaming, content creation, and AI workloads; supports CUDA 11.6.
- **Announcement Details:** Nvidia unveiled the Ada Lovelace architecture alongside the GeForce RTX 40 series GPUs on September 20, 2022.
- **Product Lineup:** GeForce RTX 4090, RTX 4080, and RTX 4070 Ti
- **Mobile and Workstation Variants:** RTX 4060, RTX 4050, and RTX 4060 Ti for laptops; RTX 6000 Ada Generation for workstations.
- **Software Support:** DLSS 3, AV1 encoding, and enhanced hardware-accelerated ray tracing.

### 13. **Hopper (2022)**

- **Release Date:** March 2022
- **Graphic Features:**
    
    - CUDA Compute Capability 8.0
    - Support for HBM2e (High Bandwidth Memory 2 enhanced)
    - PCIe 5.0
    - NVLink 3.0 for multi-GPU scaling
    - Enhanced Tensor Cores for improved AI performance
    - Memory bandwidth optimized for high-performance workloads
- **Key Features:**
    
    - Focus on AI and High-Performance Computing (HPC) applications, with specific optimizations for deep learning and scientific simulations.
    - Introduction of the new **H100 Tensor Core GPU**, offering significant performance improvements over previous generations in AI model training and inferencing.
    - Improved support for memory-intensive workloads, including large-scale AI models and simulations.
    - Enhanced support for FP64 and FP8 precision to handle the most demanding scientific and engineering computations.
    - Optimized for data centers, supercomputing, and AI-driven research.
- **Compatibility:**
    
    - Targeted at data centers, cloud computing, and AI research.
    - Supports NVIDIA software frameworks like CUDA, cuDNN, and TensorRT for deep learning workloads.
    - Enhanced multi-GPU scalability, ideal for training large models and performing parallel computing tasks.
- **Announcement Details:**
    
    - NVIDIA unveiled the Hopper architecture in March 2022, with a strong focus on revolutionizing AI and HPC workloads.
    - The architecture is named after **Grace Hopper**, a computer scientist known for her work in the development of early programming languages.
- **Product Lineup:**
    
    - The **H100 Tensor Core GPUs** are the primary product based on the Hopper architecture, specifically designed for AI, deep learning, and scientific simulations.
    - Enhanced versions of the **A100 Tensor Core** for high-performance data centers.
- **Mobile and Workstation Variants:**
    
    - While primarily targeted at data centers, it is expected that future iterations of the Hopper architecture could eventually support edge computing and mobile applications, particularly for AI workloads in autonomous systems.
- **Software Support:**
    - Supports CUDA 11.6 and later versions.
    - Optimized for machine learning frameworks such as TensorFlow, PyTorch, and Apache MXNet.
    - Includes improvements in multi-GPU setups, improving distributed training and data parallelism.
### 14. **Blackwell (2024)**

-  **Release Date:** January 30, 2025
- **Graphic Features:**
    - DirectX 12 Ultimate (Feature Level 12_2)
    - Shader Model 6.8
    - OpenCL 3.0
    - OpenGL 4.6
    - CUDA Compute Capability 9.0
    - Vulkan 1.3
    - NVENC hardware encoding (H.264, H.265, AV1)
    - Memory support with GDDR7
    - PCIe 5.0
- **Key Features:** Introduction of DLSS 4 with Multi-Frame Generation, fourth-generation RT cores, fifth-generation Tensor Cores, and support for 4:2:2 color format video encoding/decoding.
- **Compatibility:** Designed for high-performance gaming, content creation, and AI workloads; supports CUDA 12.0.
- **Announcement Details:** Nvidia announced the Blackwell architecture and GeForce RTX 50 series GPUs at CES 2025 on January 30, 2025.
- **Product Lineup:** GeForce RTX 5090, RTX 5080, RTX 5070 Ti, and RTX 5070
- **Mobile and Workstation Variants:** RTX 5060 and RTX 5050 for laptops; RTX Pro 6000 Blackwell for workstations.
- **Software Support:** DLSS 4, AV1 encoding, and enhanced hardware-accelerated ray tracing.

---

