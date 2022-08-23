# ----------
# Problem Gernerator for setting Up the problem 
# ----------


#Declare nothing function
nothingfunction(args...) = nothing;

"""
function Problem(dev::Device=CPU();
  # Numerical parameters
                nx = 64,
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
   # Declare if turn on magnetic field/VP method/Dye Module
    	     B_field = false,
         VP_method = false,
        Dye_Module = false
  # Timestepper and equation options
           stepper = "RK4",
             calcF = nothingfunction,
  # Float type and dealiasing
                 T = Float32,
  aliased_fraction = 1/3)

Construct a 3D `Problem` on device `dev`.
Keyword arguments
=================
  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.
  - `nx`: Number of grid points in ``x``-domain.
  - `ny`: Number of grid points in ``y``-domain.
  - `nz`: Number of grid points in ``z``-domain.
  - `Lx`: Extent of the ``x``-domain.
  - `Ly`: Extent of the ``y``-domain.
  - `Lz`: Extent of the ``z``-domain.
  - `ν` : Kinematic viscosity coefficient.
  - `η` : Viscosity coefficient for magnetic field.
  - `nν`: (Hyper)-viscosity order, `nν```≥ 1``, not available right now
  - `B_field` :  Declaration of B-field  
  - `VP_method`: Declaration of Volume penalization method 
  - `Dye_Module`: Declaration of Dye, Passive tracer of the flow; 
  - `stepper`: Time-stepping method.
  - `calcF`: Function that calculates the Fourier transform of the forcing, ``F̂``.
  - `aliased_fraction`: the fraction of high-wavenumbers that are zero-ed out by `dealias!()`.
  - `T`: `Float32` or `Float64`; floating point type used for `problem` data.
  - `usr_vars` : user defined variables, cloud either be emtpy or strcut
  - `usr_params` : user defined parameters, cloud either be emtpy or strcut
  - `usr_func` : user defined functions, cloud either be emtpy or strcut
"""

function Problem(dev::Device;
  # Numerical parameters
                nx = 64,
                ny = nx,
                nz = nx,
                Lx = 2π,
                Ly = Lx,
                Lz = Lx,
                dt = 0.0,
   # Drag and/or hyper-viscosity for velocity/B-field
                 ν = 0.0,
                nν = 0,
                 η = 0.0,
                nη = 0,
   # Declare if turn on magnetic field, VP method, Dye module
    	   B_field = false,
	     VP_method = false,
      Dye_Module = false,
  # Timestepper and equation options
           stepper = "RK4",
             calcF = nothingfunction,
  # Float type and dealiasing
                 T = Float32,
  aliased_fraction = 1/3,
  # User defined params/vars
          usr_vars = [],
        usr_params = [],
          usr_func = [])

  # Declare the grid
  grid = ThreeDGrid(dev, nx, Lx, ny, Ly, nz, Lz; T=T)

  # Declare vars
  vars = SetVars(dev, grid, usr_vars; B = B_field, VP =VP_method);

  # Delare params
  params = SetParams(dev, grid, calcF, usr_params; 
             B = B_field, VP = VP_method, ν = ν, η = η, nν = nν);

  # Declare Fiuld Equations that will be iterating 
  equation = Equation_with_forcing(dev, grid; B = B_field, VP = VP_method);

  # Return the Problem
  return MHDFLowsProblem(equation, stepper, dt, grid, vars, params, dev;
          BFlag = B_field, VPFlag = VP_method, DyeFlag = Dye_Module, usr_func = usr_func)

end

function Equation_with_forcing(dev, grid::AbstractGrid; B = false, VP= false)
  T = eltype(grid);
  Nₗ = ifelse(B,6,3)
  L = zeros(dev, T, (grid.nkr, grid.nl, grid.nm, Nₗ));

  if (B)
  	calcN! = ifelse(VP,MHDcalcN_VP!,MHDcalcN!);
  else
  	calcN! = ifelse(VP, HDcalcN_VP!, HDcalcN!);
  end
  
  return FourierFlows.Equation(L,calcN!, grid);
end


function MHDcalcN!(N, sol, t, clock, vars, params, grid)
  dealias!(sol, grid)
  
  MHDSolver.MHDcalcN_advection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function HDcalcN!(N, sol, t, clock, vars, params, grid)
  dealias!(sol, grid)
  
  HDSolver.HDcalcN_advection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function MHDcalcN_VP!(N, sol, t, clock, vars, params, grid)
  dealias!(sol, grid)
  
  MHDSolver_VP.MHDcalcN_advection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function HDcalcN_VP!(N, sol, t, clock, vars, params, grid)
  dealias!(sol, grid)
  
  HDSolver_VP.HDcalcN_advection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function addforcing!(N, sol, t, clock, vars, params, grid)
  params.calcF!(N, sol, t, clock, vars, params, grid) 
  return nothing
end
