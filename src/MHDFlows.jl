module MHDFlows

using 
  CUDA,
  Statistics,
  SpecialFunctions,
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

include("pgen.jl")
include("MHDSolver.jl")
include("HDSolver.jl")
include("datastructure.jl")
include("integrator.jl")

@reexport using MHDFlows.pgen
@reexport using MHDFlows.MHDSolver
@reexport using MHDFlows.HDSolver
@reexport using MHDFlows.datastructure
@reexport using MHDFlows.integrator

end