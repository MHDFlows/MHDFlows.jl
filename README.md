# MHDFlows.jl

Three Dimensional Magnetohydrodynamic(MHD) pseudospectral solvers written in julia language with <a href="http://github.com/FourierFlows/FourierFlows.jl">FourierFlows.jl</a>. 

This package leaverages the [FourierFlows.jl](http://github.com/FourierFlows/FourierFlows.jl) package to set up a module in order to solve the small 3D incompessbile MHD problems in periodic domains using pesudespectral method. Feel free to modify yourself for your own research prepuse

## Version No.
Beta 1.0

## Installation Guide & compatibility 
Currently, you can only download and bluid it yourself. A Julia's built-in package manager installation will be avaibile after the stable release update.

The current version is tested in both Julia v1.2 and v1.7 version.

## Scalability 
The MHD Solver could either run on CPU or GPU. The scalability is same as Fourierflows, which is restricted to either a single CPU or single GPU. This restriction may change in the future depending on the development of FourierFlows. If you running the package usiing GPU, the CUDA package is needed for operation. Check out [CUDA.jl](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#Device-Management) for more detail. 

**Important note**: As this is a 3D solver, the memory uasge and computational time scale as `N^3`. Beware of the memory usage aspecially when you are using the GPU. 

## Example
Two examples were set up to illustrate the workflow of using this package. See `example\` for more detail. 

## Developer
MHDFlows is developed by [Ka Wai HO@UW-Madison Astronomy](https://scholar.google.com/citations?user=h2j8wbYAAAAJ&hl=en).

## Citing
A paper can be cited elsewhere in the future : ). Feel free to cite the github page right now. 
