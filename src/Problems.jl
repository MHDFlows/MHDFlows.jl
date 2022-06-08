# ----------
# Modified Problem Struct from FourierFlows
# ----------


"""
    struct Equation{T, TL, G<:AbstractFloat}
    
The equation to be solved `∂u/∂t = L*u + N(u)`. Array `L` includes the coefficients
of the linear term `L*u` and `calcN!` is a function which computes the nonlinear
term `N(u)`. The struct also includes the problem's `grid` and the float type of the
state vector (and consequently of `N(u)`).
$(TYPEDFIELDS)
"""
struct Equation{T, TL, G<:AbstractFloat}
    "array with coefficient for the linear part of the equation"
       L :: TL
    "function that computes the nonlinear part of the equation"
  calcN! :: Function
    "the grid"
    grid :: AbstractGrid{G}
    "the dimensions of `L`"
    dims :: Tuple
    "the float type for the state vector"
       T :: T # eltype or tuple of eltypes of sol and N
end

"""
    Equation(L, calcN!, grid; dims=supersize(L), T=nothing)
    
The equation constructor from the array `L` of the coefficients of the linear term, the function 
`calcN!` that computes the nonlinear term and the `grid` for the problem.
"""
function Equation(L, calcN!, grid::AbstractGrid{G}; dims=supersize(L), T=nothing) where G
  T = T == nothing ? T = cxtype(G) : T
  
  return Equation(L, calcN!, grid, dims, T)
end

CheckON(Flag_equal_to_True::Bool) = Flag_equal_to_True ? string("ON") : string("OFF")

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
    "Volume Penalization"
    vp :: Bool
end

"""
    struct MHDFLowsProblem{T, A<:AbstractArray, Tg<:AbstractFloat, TL}
    
A problem that represents a partial differential equation.
$(TYPEDFIELDS)
"""
struct MHDFlowsProblem{T, A<:AbstractArray, Tg<:AbstractFloat, TL, Dye, usr_foo} <: AbstractProblem
    "the state vector"
          sol :: A
    "the problem's slock"
        clock :: FourierFlows.Clock{Tg}
    "the equation"
          eqn :: FourierFlows.Equation{T, TL, Tg}
    "the grid"
         grid :: AbstractGrid{Tg}
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
                 BFlag = false, VPFlag = false, DyeFlag = false, usr_func = [],
                 stepperkwargs...) where T
  clock = FourierFlows.Clock{T}(dt, 0, 0)

  timestepper = FourierFlows.TimeStepper(stepper, eqn, dt, dev);

  sol = FourierFlows.devzeros(dev, eqn.T, eqn.dims);

  flag = Flag(BFlag, VPFlag);

  dye = DyeContructer(dev, DyeFlag, grid);

  usr_func = length(usr_func) == 0 ? [nothingfunction] : usr_func;
  return MHDFlowsProblem(sol, clock, eqn, grid, vars, params, timestepper, flag, usr_func, dye)
end

show(io::IO, problem::MHDFlowsProblem) =
    print(io, "MHDFlows Problem\n",
    	      "  │    Funtions\n",
    		  "  │     ├──────── B-field: "*CheckON(problem.flag.b),'\n',
    		  "  ├─────├────── VP Method: "*CheckON(problem.flag.vp),'\n',
    		  "  │     ├──────────── Dye: "*CheckDye(problem.dye),'\n',
    		  "  │     └── user function: "*CheckFunction(problem.usr_func),'\n',
    		  "  │                        ",'\n',
              "  │     Features           ",'\n',  
              "  │     ├─────────── grid: grid (on " * FourierFlows.griddevice(problem.grid) * ")", '\n',
              "  │     ├───── parameters: params", '\n',
              "  │     ├────── variables: vars", '\n',
              "  └─────├─── state vector: sol", '\n',
              "        ├─────── equation: eqn", '\n',
              "        ├────────── clock: clock", '\n',
              "        └──── timestepper: ", string(nameof(typeof(problem.timestepper))))
