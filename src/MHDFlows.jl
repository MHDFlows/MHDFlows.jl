module MHDFlows

using 
  CUDA,
  Statistics,
  SpecialFunctions,
  Reexport,
  DocStringExtensions,
  HDF5,
  FFTW

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
import Base: show, summary

"Abstract supertype for problem."
abstract type AbstractProblem end
abstract type MHDVars <: AbstractVars end

include("DyeModule.jl")
include("Problems.jl")
include("pgen.jl")
include("Solver/HDSolver.jl")
include("Solver/MHDSolver.jl")
include("Solver/HDSolver_VP.jl")
include("Solver/MHDSolver_VP.jl")
include("DiagnosticWrapper.jl")
include("integrator.jl")
include("datastructure.jl")
include("utils/VectorCalculus.jl")
include("utils/MHDAnalysis.jl")
include("utils/GeometryFunction.jl")
include("utils/func.jl")


#pgen module
include("pgen/A99ForceDriving.jl")
include("pgen/TaylorGreenDynamo.jl")

export Problem,           
       TimeIntegrator!,
       Restart!,
       Cylindrical_Mask_Function,
       SetUpProblemIC!,
       Curl,            
       Div,
       LaplaceSolver,
       Crossproduct,
       Dotproduct,
       ∂i,∇X,
       xy_to_polar,       
       ScaleDecomposition, 
       h_k,
       h_m,
       VectorPotential,
       LaplaceSolver,
       getL,
       spectralline,
       ⋅, ×
end
