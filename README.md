# MHDFlows.jl
![Julia flow](img/TG_Instability.gif)

Three Dimensional Magnetohydrodynamic(MHD) pseudospectral solvers written in Julia language with <a href="http://github.com/FourierFlows/FourierFlows.jl">FourierFlows.jl</a>. This solver support the following features:

1. 2/3D incompressible HD/MHD simulation (periodic boundary)
3. Incompressible  HD/MHD simulation with volume penalization method
4. Isothermal compressible  HD/MHD simulation (periodic boundary)
5. 2/3D Electron magnetohydrodynamic simulation (periodic boundary)
6. Passive Dye Tracer (Experimental Feature)

This package leverages the [FourierFlows.jl](http://github.com/FourierFlows/FourierFlows.jl) package to set up the module. The main purpose of MHDFlows.jl aims to solve the portable 3D MHD problems on personal computer instead of cluster. Utilizing the Nvidia CUDA technology, the MHDFlows.jl could solve the front-end MHD turbulence research problems in the order of few-ten minutes by using a mid to high end gaming display card (see Memory usage & speed section). Feel free to modify yourself for your own research purpose.

## Version No.
v 0.2.0  
note : v 0.2.0 will be the final major update before the multi-gpu version release 

## Installation Guide & compatibility 
The current version is tested on v1.7.3/1.8.2/1.9.0 version.

Currently, you have two way of installing MHDFlows.jl

1. Download and build it yourself from here. 

2. Julia's built-in package manager installation (accessed by pressing `]` in the Julia REPL command prompt)

   ```julia
   julia>
   (v1.8) pkg> add MHDFlows
   ```


## Scalability 
The MHD Solver could either run on CPU or GPU. The scalability is same as Fourierflows, which is restricted to either a single CPU or single GPU. This restriction may change in the future depending on the development of FourierFlows. If you are running the package using GPU, the CUDA package is needed for the computation. Check out [CUDA.jl](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#Device-Management) for more detail. 

**Important note**: As this is a 3D solver, the memory usage and computational time scale as `N^3` (at least). Beware of the memory usage especially when you are using the GPU. 

## Memory usage & speed

**Memory usage**

For GPU users, here are some useful numbers of memory requirement for choosing the resolution of the simulation with RK4/ LSRK4 method. You may end up getting higher resolution for the same memory.

| Memory Size | Maximum Resolution ( $N^3$ )  |
| ----------- | ------------------------------|
| 6 GB        | $256^3$ (pure MHD simulation) |
| 10 GB       | $320^3$ (pure MHD simulation) |
| 24 GB       | $512^3$ (pure MHD simulation) |
| 80 GB       | $700^3$ (pure MHD simulation) |

**Speed**

The following table provides the reference of the average runtime of 1 iteration in pure HD/MHD computation. As the benchmarks are running on the WSL2, the runtime could varies and does not reflect the best performance.

Method: compute the average time used of 100 iterations using RK4 method

Environment: WSL2 in Win11 (Ubuntu 18.04/20.04 LTS through jupyter-lab)

**HD** (Taylor Green Vortex, T = Float32)

| Spec CPU/GPU                | $32^3$ | $64^3$ | $128^3$ | $256^3$ |
| --------------------------- | ------ | ------ | ------- | ------- |
| AMD Ryzen 7 5800x 8 threads | 0.040s | 0.074s | 0.490S  | 7.025s  |
| NVIDIA RTX 3080 10GB        | 0.016s | 0.018s | 0.023s  | 0.152s  |

**MHD** (Taylor Green Vortex, T = Float32)

| Spec CPU/GPU                | $32^3$ | $64^3$ | $128^3$ | $256^3$ |
| --------------------------- | ------ | ------ | ------- | ------- |
| AMD Ryzen 7 5800x 8 threads | 0.049s | 0.180s | 1.490s  | 14.50s  |
| NVIDIA RTX 3080 10GB        | 0.013s | 0.012s | 0.037s  | 0.271s  |

## Example
Few examples were set up to illustrate the workflow of using this package. [Check out](https://github.com/MHDFlows/MHDFlows-Example) for more detail.  The documentation is work in progress and will be available in the future. 

## Developer
MHDFlows is currently developed by [Ka Wai HO@UW-Madison Astronomy](https://scholar.google.com/citations?user=h2j8wbYAAAAJ&hl=en).

## Citing
A paper can be cited elsewhere in the future :slightly_smiling_face:. Feel free to cite the GitHub page right now. 
