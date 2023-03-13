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
              EMHD = false,
   Compressibility = false,
             Shear = false,
         VP_method = false,
        Dye_Module = false,
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
  - `EMHD` : Declarartion of E-MHD
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
  # Numerical & physical parameters
                nx = 64,
                ny = nx,
                nz = nx,
                Lx = 2π,
                Ly = Lx,
                Lz = Lx,
                cₛ = 0.0,
                dt = 0.0,
   # Drag and/or hyper-viscosity for velocity/B-field
                 ν = 0.0,
                nν = 0,
                 η = 0.0,
                nη = 0,
   # Declare if turn on magnetic field, EMHD, VP method, Dye module
         B_field = false,
            EMHD = false,
 Compressibility = false,
           Shear = false,
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

  # Compatibility Checking
  if cₛ == 0.0 && Compressibility
    error("You should define cₛ")
  end

  # Declare the grid
  if Shear
    error("Shear haven't fully implemented yet!")
    grid = GetShearingThreeDGrid(dev; nx=nx, Lx=Lx, ny = ny, Ly = Ly, nz = nz, Lz = Lz, T=T)
  else
    grid = ThreeDGrid(dev; nx=nx, Lx=Lx, ny = ny, Ly = Ly, nz = nz, Lz = Lz, T=T)
  end

  # Declare vars
  vars = SetVars(dev, grid, usr_vars; B = B_field, E = EMHD, VP = VP_method, C = Compressibility);

  # Delare params
  params = SetParams(dev, grid, calcF, usr_params; 
             B = B_field, E = EMHD, VP = VP_method, C= Compressibility, S=Shear,
             cₛ = cₛ, ν = ν, η = η, nν = nν);

  # Declare Fiuld Equations that will be iterating 
  equation = Equation_with_forcing(dev, grid; B = B_field, E = EMHD, C = Compressibility, S=Shear);

  # Return the Problem
  return MHDFLowsProblem(equation, stepper, dt, grid, vars, params, dev;
          CFlag = Compressibility, BFlag = B_field, EFlag = EMHD, SFlag = Shear, 
          VPFlag = VP_method, DyeFlag = Dye_Module, 
          usr_func = usr_func)

end

function Equation_with_forcing(dev, grid; B = false, E = false, C = false, S=false)
  if C 
    Nₗ = ifelse(B,7,4)
  else
    if E
      Nₗ = 3
    else
      Nₗ = ifelse(B,6,3)
    end
  end
  if C
    calcN! = B ? CMHDcalcN! : CHDcalcN!
  elseif S
    calcN! = B ? SMHDcalcN! : SHDcalcN!
  elseif E
    calcN! = EMHDcalcN!
  else
    calcN! = B ? MHDcalcN! : HDcalcN!
  end
  
  return Setup_Equation(calcN!, grid; Nl =Nₗ)
end


function MHDcalcN!(N, sol, t, clock, vars, params, grid)
  
  dealias!(sol, grid)
  
  MHDSolver.MHDcalcN_advection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function EMHDcalcN!(N, sol, t, clock, vars, params, grid)
  
  dealias!(sol, grid)
  
  MHDSolver.EMHDcalcN_advection!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function HDcalcN!(N, sol, t, clock, vars, params, grid)
  dealias!(sol, grid)

  addforcing!(N, sol, t, clock, vars, params, grid)
  
  HDSolver.HDcalcN_advection!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function SMHDcalcN!(N, sol, t, clock, vars, params, grid)
  
  Shear.Shearing_dealias!(sol, grid);

  #Shear.Shearing_coordinate_update!(N, sol, t, clock, vars, params, grid)
  
  Shear.MHD_ShearingAdvection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function SHDcalcN!(N, sol, t, clock, vars, params, grid)
  
  Shear.Shearing_dealias!(sol, grid);

  Shear.Shearing_coordinate_update!(N, sol, t, clock, vars, params, grid);
  
  Shear.HD_ShearingAdvection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function CMHDcalcN!(N, sol, t, clock, vars, params, grid)
  
  dealias!(sol, grid)
  
  MHDSolver_compressible.MHDcalcN_advection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function CHDcalcN!(N, sol, t, clock, vars, params, grid)
  dealias!(sol, grid)
  
  HDSolver_compressible.HDcalcN_advection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end


function addforcing!(N, sol, t, clock, vars, params, grid)
  params.calcF!(N, sol, t, clock, vars, params, grid) 
  return nothing
end