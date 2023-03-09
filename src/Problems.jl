# ----------
# Modified Problem Struct from FourierFlows
# ----------


"""
    Equation(L, calcN!, grid; dims=supersize(L), T=nothing)
    
The equation constructor from the array `L` of the coefficients of the linear term, the function 
`calcN!` that computes the nonlinear term and the `grid` for the problem.
"""
function Setup_Equation(calcN!, grid::AbstractGrid{G}; T=nothing, Nl = 3) where G
  dims = tuple(size(grid.Krsq)...,Nl)
  T = T == nothing ? T = cxtype(G) : T
  #Compatibility to FourierFlows.Equation 
  L = 0
  return FourierFlows.Equation(L, calcN!, grid; dims=dims)
end

CheckON(Flag_equal_to_True::Bool) = Flag_equal_to_True ? string("ON") : string("OFF")

function CheckON(FlagB::Bool, FlagE::Bool) 
  if FlagB 
    FlagE ? string("ON (EMHD)") : string("ON (Ideal MHD)")
  else
    string("OFF")
  end

end

function CheckDye(dye::Dye)
    if dye.dyeflag == true
        return string("ON, at prob.dye")
    else
        return string("OFF")
    end
end 

function CheckFunction(func)
    if length(func) > 0 && func[1] != nothingfunction;
        return string("ON, Number of function = "*string(length(func)));
    else
        return string("OFF")
    end
end

"""
    mutable struct Clock{T<:AbstractFloat}
    
Represents the clock of a problem.
$(TYPEDFIELDS)
"""
mutable struct Clock{T<:AbstractFloat}
    "the time-step"
    dt :: T
    "the time"
     t :: T
    "the step number"
  step :: Int
end

"""
    mutable struct Flag
    
Represents the Flag of a problem.
$(TYPEDFIELDS)
"""
struct Flag
    "Magnetic Field"
     b :: Bool
     "EMHD Field"
     e :: Bool
    "Volume Penalization"
    vp :: Bool
    "Compressibility"
     c :: Bool
     "Shear"
     s :: Bool
end

"""
    struct MHDFLowsProblem{T, A<:AbstractArray, Tg<:AbstractFloat, TL}
    
A problem that represents a partial differential equation.
$(TYPEDFIELDS)
"""
struct MHDFlowsProblem{T, A<:AbstractArray, Tg<:AbstractFloat, TL, Dye, usr_foo, AbstractGrid} <: AbstractProblem
    "the state vector"
          sol :: A
    "the problem's slock"
        clock :: FourierFlows.Clock{Tg}
    "the equation"
          eqn :: FourierFlows.Equation{T, TL, Tg}
    "the grid"
         grid :: AbstractGrid
    "the variables"
         vars :: AbstractVars
    "the parameters"
       params :: AbstractParams
    "the timestepper"
  timestepper :: AbstractTimeStepper{A}
    "the flag for B-field and Volume Penalization Method"
         flag :: Flag
    "the user defined function"
     usr_func :: usr_foo
    "the Dye module"
          dye :: Dye     
end

"""
    MHDFLowsProblem(eqn::Equation, stepper, dt, grid::AbstractGrid{T}, 
            vars=EmptyVars, params=EmptyParams, dev::Device=CPU(); stepperkwargs...) where T
Construct a `Problem` for equation `eqn` using the time`stepper` with timestep 
`dt`, on `grid` and on `dev`ice. Optionally, use the keyword arguments to provide 
variables with `vars` and parameters with `params`. The `stepperkwargs` are passed
to the time-stepper constructor.
"""
function MHDFLowsProblem(eqn::FourierFlows.Equation, stepper, dt, grid::AbstractGrid{T}, 
                 vars=EmptyVars, params=EmptyParams, dev::Device=CPU(); 
                 BFlag = false, EFlag = false, VPFlag = false, CFlag = false, SFlag = false, DyeFlag = false, usr_func = [],
                 stepperkwargs...) where T

  clock = FourierFlows.Clock{T}(dt, 0, 0)
  if EFlag && stepper == "HM89"
  #  timestepper = eSSPIFRK3TimeStepper(eqn, dev) #For SFlag
    timestepper = HM89TimeStepper(eqn, dev) #For EFlag
  else
    timestepper = FourierFlows.TimeStepper(stepper, eqn, dt, dev)
  end
  sol = zeros(dev, eqn.T, eqn.dims)

  flag = Flag(BFlag, EFlag, VPFlag, CFlag, SFlag)

  dye = DyeContructer(dev, DyeFlag, grid)

  usr_func = length(usr_func) == 0 ? [nothingfunction] : usr_func

  return MHDFlowsProblem(sol, clock, eqn, grid, vars, params, timestepper, flag, usr_func, dye)

end

show(io::IO, problem::MHDFlowsProblem) =
    print(io, "MHDFlows Problem\n",
    	    "  │    Funtions\n",
          "  │     ├ Compressibility: "*CheckON(problem.flag.c),'\n',
          "  │     ├──────── B-field: "*CheckON(problem.flag.b,problem.flag.e),'\n',
          "  │     ├────────── Shear: "*CheckON(problem.flag.s),'\n',
    		  "  ├─────├────── VP Method: "*CheckON(problem.flag.vp),'\n',
    		  "  │     ├──────────── Dye: "*CheckDye(problem.dye),'\n',
    		  "  │     └── user function: "*CheckFunction(problem.usr_func),'\n',
    		  "  │                        ",'\n',
          "  │     Features           ",'\n',  
          "  │     ├─────────── grid: grid (on " * string(typeof(problem.grid.device)) * ")", '\n',
          "  │     ├───── parameters: params", '\n',
          "  │     ├────── variables: vars", '\n',
          "  └─────├─── state vector: sol", '\n',
          "        ├─────── equation: eqn", '\n',
          "        ├────────── clock: clock", '\n',
          "        └──── timestepper: ", string(nameof(typeof(problem.timestepper))))