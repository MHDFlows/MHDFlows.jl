module pgen
#Problem Gernerator for setting Up the problem 

using 
  CUDA,
  Statistics,
  SpecialFunctions,
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

include("MHDSolver.jl")
include("HDSolver.jl")
include("datastructure.jl")

export MHDProblem,
       HDProblem,
       MHDcalcN,
       HDcalcN


MHDParams  = datastructure.MHDParams;
HDParams   = datastructure.HDParams;
SetMHDVars = datastructure.SetMHDVars;
SetHDVars  = datastructure.SetHDVars;
MHDcalcN_advection!  = MHDSolver.MHDcalcN_advection!;
HDcalcN_advection!   = HDSolver.HDcalcN_advection!;

nothingfunction(args...) = nothing;


"""
function MHDProblem(dev::Device=CPU();
  # Numerical parameters
                nx = 6s4,
                ny = nx,
                nz = nx,
                Lx = 2π,
                Ly = Lx,
                Lz = Lx,
   # Drag and/or hyper-viscosity for velocity/B-field
                 ν = 0,
                nν = 1,
                 μ = 0,
                 η = 0,
                nμ = 0,
  # Timestepper and equation options
                dt = 0.01,
           stepper = "RK4",
             calcF = nothingfunction,
  # Float type and dealiasing
                 T = Float32)

Construct a 3D MHD `problem` on device `dev`.
Keyword arguments
=================
  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.
  - `nx`: Number of grid points in ``x``-domain.
  - `ny`: Number of grid points in ``y``-domain.
  - `Lx`: Extent of the ``x``-domain.
  - `Ly`: Extent of the ``y``-domain.
  - `Lz`: Extent of the ``z``-domain.
  - `ν` : Kinematic viscosity coefficient.
  - `nν`: (Hyper)-viscosity order, `nν```≥ 1``, not available right now
  - `η` : Viscosity coefficient for magnetic field.
  - `dt`: Time-step.
  - `stepper`: Time-stepping method.
  - `calcF`: Function that calculates the Fourier transform of the forcing, ``F̂``.
  - `aliased_fraction`: the fraction of high-wavenumbers that are zero-ed out by `dealias!()`.
  - `T`: `Float32` or `Float64`; floating point type used for `problem` data.
"""

function MHDProblem(dev::Device=CPU();
  # Numerical parameters
                nx = 6s4,
                ny = nx,
                nz = nx,
                Lx = 2π,
                Ly = Lx,
                Lz = Lx,
   # Drag and/or hyper-viscosity for velocity/B-field
                 ν = 0,
                nν = 1,
                 η = 0,
  # Timestepper and equation options
                dt = 0.01,
           stepper = "RK4",
             calcF = nothingfunction,
  # Float type and dealiasing
                 T = Float32)

  grid = ThreeDGrid(dev, nx, Lx, ny, Ly, nz, Lz; T=T)

  params = MHDParams{T}(ν, η, nν, 1, 2, 3, 4, 5, 6,calcF)

  vars = SetMHDVars(dev, grid);

  equation = Equation_with_forcing(dev, params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)

end



"""
function HDProblem(dev::Device=CPU();
  # Numerical parameters
                nx = 64,
                ny = nx,
                nz = nx,
                Lx = 2π,
                Ly = Lx,
                Lz = Lx,
   # Drag and/or hyper-viscosity for velocity
                 ν = 0,
                nν = 1,
                 μ = 0,
   # force function 
                calcF = nothingfunction,
  # Timestepper and equation options
                dt = 0.01,
           stepper = "RK4",
  # Float type and dealiasing
                 T = Float32)

Construct a 3D HD `problem` on device `dev`.
Keyword arguments
=================
  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.
  - `nx`: Number of grid points in ``x``-domain.
  - `ny`: Number of grid points in ``y``-domain.
  - `Lx`: Extent of the ``x``-domain.
  - `Ly`: Extent of the ``y``-domain.
  - `Lz`: Extent of the ``z``-domain.
  - `ν` : Kinematic viscosity coefficient.
  - `nν`: (Hyper)-viscosity order, `nν```≥ 1``, not available right now
  - `dt`: Time-step.
  - `stepper`: Time-stepping method.
  - `calcF`: Function that calculates the Fourier transform of the forcing, ``F̂``.
  - `aliased_fraction`: the fraction of high-wavenumbers that are zero-ed out by `dealias!()`.
  - `T`: `Float32` or `Float64`; floating point type used for `problem` data.
"""

function HDProblem(dev::Device=CPU();
  # Numerical parameters
                nx = 64,
                ny = nx,
                nz = nx,
                Lx = 2π,
                Ly = Lx,
                Lz = Lx,
   # Drag and/or hyper-viscosity for velocity
                 ν = 0,
                nν = 1,
   # force function 
                calcF = nothingfunction,
  # Timestepper and equation options
                dt = 0.01,
           stepper = "RK4",
  # Float type and dealiasing
                 T = Float32)

  grid = ThreeDGrid(dev, nx, Lx, ny, Ly, nz, Lz; T=T)

  params = HDParams{T}(ν, nν, 1, 2, 3, calcF)

  vars = SetHDVars(dev, grid);

  equation = Equation_with_forcing(dev,params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
end


function MHDcalcN!(N, sol, t, clock, vars, params, grid)
  dealias!(sol, grid)
  
  MHDcalcN_advection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function HDcalcN!(N, sol, t, clock, vars, params, grid)
  dealias!(sol, grid)
  
  HDcalcN_advection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function Equation_with_forcing(dev,params::MHDParams, grid::AbstractGrid)
  T = eltype(grid)
  L = zeros(dev, T, (grid.nkr,grid.nm,grid.nk, 6));
    
  return FourierFlows.Equation(L,MHDcalcN!, grid)
end

function Equation_with_forcing(dev,params::HDParams, grid::AbstractGrid)
  
  T = eltype(grid)
  L = zeros(dev, T, (grid.nkr,grid.nm,grid.nk, 6));
    
  return FourierFlows.Equation(L,HDcalcN!, grid)
end

function addforcing!(N, sol, t, clock, vars, params, grid)
  
  params.calcF!(N, sol, t, clock, vars, params, grid)

  return nothing

end


end