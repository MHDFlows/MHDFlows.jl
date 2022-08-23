# MHDFlows.jl
![Julia flow](img/TG_Instability.gif)

Three Dimensional Magnetohydrodynamic(MHD) pseudospectral solvers written in Julia language with <a href="http://github.com/FourierFlows/FourierFlows.jl">FourierFlows.jl</a>. This solver support the following features:

1. 2D incompressible HD/MHD simulation (periodic boundary)
2. 3D incompressible HD/MHD simulation (periodic boundary)
3. Incompressible  HD/MHD simulation with volume penalization method
4. Passive Dye Tracer (Experimental Feature)

This package leverages the [FourierFlows.jl](http://github.com/FourierFlows/FourierFlows.jl) package to set up the module. The main purpose of MHDFlows.jl aims to solve the portable 3D MHD problems on personal computer instead of cluster. Utilizing the Nvidia CUDA technology, the MHDFlows.jl could solve the front-end MHD turbulence problem in the order of few-ten minutes by using a mid to high end gaming display card (see Memory usage & speed section). Feel free to modify yourself for your own research purpose.

## Version No.
v 0.1.0

## Installation Guide & compatibility 
The current version is tested on v1.5.3/1.7.3/1.8 version.

Currently, you have two way of installing MHDFlows.jl

1. Download and build it yourself from here. 

2. Julia's built-in package manager installation (accessed by pressing `]` in the Julia REPL command prompt)

   ```julia
   julia>
   (v1.7) pkg> add MHDFlows
   ```



## Scalability 
The MHD Solver could either run on CPU or GPU. The scalability is same as Fourierflows, which is restricted to either a single CPU or single GPU. This restriction may change in the future depending on the development of FourierFlows. If you are running the package using GPU, the CUDA package is needed for the computation. Check out [CUDA.jl](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#Device-Management) for more detail. 

**Important note**: As this is a 3D solver, the memory usage and computational time scale as `N^3` (at least). Beware of the memory usage especially when you are using the GPU. 

## Memory usage & speed

**Memory usage**

For GPU users, here are some useful numbers of memory requirement for choosing the resolution of the simulation. You may end up getting higher resolution for the same memory.

| Memory Size | Maximum Resolution ($N^3$ )    |
| ----------- | ------------------------------ |
| 6 GB        | $256^3$ (pure HD simulation) |
| 10 GB       | $300^3$ (pure MHD simulation) |

**Speed**

The following table provides the reference of the runtime for 1 iteration in pure HD/MHD computation. As the benchmarks are running on the WSL2, the runtime could varies and does not reflect the best performance.

Method: compute the mean time of 20 iterations using RK4 method

Environment: WSL2 in Win11 (Ubuntu 18.04 LTS through jupyter-lab)

**HD** (Taylor Green Vortex)

| Spec CPU/GPU                | $32^3$ | $64^3$ | $128^3$ | $256^3$ |
| --------------------------- | ------ | ------ | ------- | ------- |
| AMD Ryzen 7 5800x 8 threads | 0.139s | 0.178s | 0.764s  | 7.025s  |
| NVIDIA RTX 3080 10GB        | 0.016s | 0.018s | 0.038s  | 0.211s  |

**MHD** (Taylor Green Vortex)

| Spec CPU/GPU                | $32^3$ | $64^3$ | $128^3$ | $256^3$ |
| --------------------------- | ------ | ------ | ------- | ------- |
| AMD Ryzen 7 5800x 8 threads | 0.19s  | 0.231s | 1.8s    | 18.48s  |
| NVIDIA RTX 3080 10GB        | 0.041s | 0.060s | 0.15s   | 1.23s   |

## Example
Few examples were set up to illustrate the workflow of using this package. See `example\` for more detail. 

## Developer
MHDFlows is developed by [Ka Wai HO@UW-Madison Astronomy](https://scholar.google.com/citations?user=h2j8wbYAAAAJ&hl=en).

## Citing
A paper can be cited elsewhere in the future :slightly_smiling_face:. Feel free to cite the GitHub page right now. 
