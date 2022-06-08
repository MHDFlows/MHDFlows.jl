# MHDFlows.jl
![Julia flow](img/TG_Instability.gif)

Three Dimensional Magnetohydrodynamic(MHD) pseudospectral solvers written in Julia language with <a href="http://github.com/FourierFlows/FourierFlows.jl">FourierFlows.jl</a>. This solver support the following features:

1. 2D incompressible  HD/MHD simulation (periodic boundary)
2. 3D incompressible  HD/MHD simulation (periodic boundary)
3. incompressible  HD/MHD simulation with volume penalization method (Public version not released yet)
4. Passive Dye Tracer (Experimental Feature)

This package leverages the [FourierFlows.jl](http://github.com/FourierFlows/FourierFlows.jl) package to set up a module in order to solve the portable 3D incompressible MHD problems in periodic domains using pseudo-spectral method. Feel free to modify yourself for your own research purpose.

## Version No.
Beta 3.0

## Installation Guide & compatibility 
Currently, you can only download and build it yourself. A Julia's built-in package manager installation will be available after the stable release update.

The current version is tested in v1.7.2/1.7.3 version.

## Scalability 
The MHD Solver could either run on CPU or GPU. The scalability is same as Fourierflows, which is restricted to either a single CPU or single GPU. This restriction may change in the future depending on the development of FourierFlows. If you are running the package using GPU, the CUDA package is needed for the computation. Check out [CUDA.jl](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#Device-Management) for more detail. 

**Important note**: As this is a 3D solver, the memory usage and computational time scale as `N^3`. Beware of the memory usage especially when you are using the GPU. 

## Memory usage & speed

**Memory usage**

For GPU users, here are some useful numbers of memory requirement for choosing the resolution of the simulation. You may end up getting higher resolution for the same memory.

| Memory Size | Maximum Resolution ($N^3$ )    |
| ----------- | ------------------------------ |
| 6 GB        | $256^3 $ (pure HD simulation)  |
| 10 GB       | $300^3 $ (pure MHD simulation) |

**Speed**

The following table provides the reference of the runtime for 1 iteration in pure HD/MHD computation.

Method : compute the mean time of 20 iteration using RK4 method

Environment: WSL2 in Win11 (Ubuntu 18.04 LTS) ( in jupyter-lab)

**HD** (Taylor Green Vortex)

| Spec CPU/GPU                | $32^3$ | $64^3$ | $128^3$ | $256^3$ |
| --------------------------- | ------ | ------ | ------- | ------- |
| AMD Ryzen 7 5800x 8 threads | 0.139s | 0.178s | 0.764s  | 7.025s  |
| NVIDIA RTX 3080 10GB        | 0.016s | 0.018s | 0.038s  | 0.211s  |

**MHD** (Taylor Green Vortex)

| Spec CPU/GPU                | $32^3$ | $64^3$ | $128^3$ | $256^3$ |
| --------------------------- | ------ | ------ | ------- | ------- |
| AMD Ryzen 7 5800x 8 threads | 0.330s | 0.338s | 2.699s  | 29.21s  |
| NVIDIA RTX 3080 10GB        | 0.041s | 0.060s | 0.282s  | 2.133s  |

## Example
Few examples were set up to illustrate the workflow of using this package. See `example\` for more detail. 

## Developer
MHDFlows is developed by [Ka Wai HO@UW-Madison Astronomy](https://scholar.google.com/citations?user=h2j8wbYAAAAJ&hl=en).

## Citing
A paper can be cited elsewhere in the future : ). Feel free to cite the GitHub page right now. 
