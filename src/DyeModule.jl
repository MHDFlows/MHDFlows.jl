# ----------
# Dye Module, Experimental feature!
# ----------

"""
    mutable struct Dye
    
Represents the clock of a problem.
$(TYPEDFIELDS)
"""
mutable struct Dye{Bool,Function,Array,Tmp}
    "Dye Flag"
     dyeflag :: Bool
    "Dye Function"
stepforward! :: Function
    "Dye Array in complexed Form"
           ρ :: Array
    "Dye Sol and cache"
         tmp :: Tmp
end

show(io::IO, dye::Dye) =
     print(io, "Dye\n",
               "  ├────── dye flag: ", dye.dyeflag, '\n',
               "  ├────── function: ", dye.stepforward!, '\n',
               "  └─────────── dye: ρ - size :", size(dye.ρ));

mutable struct Tmp{Atrans}
  sol₀ :: Atrans
  sol₁ :: Atrans
  RHS₁ :: Atrans
  RHS₂ :: Atrans
  RHS₃ :: Atrans
  RHS₄ :: Atrans
end

function DyeContructer(::Dev, DyeFlag::Bool, grid::AbstractGrid) where Dev
  T = eltype(grid)
  
  if (DyeFlag == true )
  	@devzeros Dev T (grid.nx, grid.ny, grid.nz) ρ
  	@devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm) sol₀ sol₁ RHS₁ RHS₂ RHS₃ RHS₄
  	tmp = Tmp(sol₀, sol₁, RHS₁, RHS₂, RHS₃, RHS₄);
  	dye = Dye(true, Dyestepforward!, ρ, tmp);
  else
  	ρ  = zeros(T,1);
  	tmp = Tmp(zeros(1),zeros(1),zeros(1),zeros(1),zeros(1),zeros(1));
  	dye = Dye(false, Dyestepforward!, ρ, tmp);
  end
  return dye
end

function Dyestepforward!(prob)
  dye, clock, vars, params, grid, clock = prob.dye, prob.clock, prob.vars, prob.params, prob.grid, prob.clock;
  ts = dye.tmp;
  RK4substeps!(ts.sol₀, clock, dye.tmp, vars, params, grid, clock.t, clock.dt)
  RK4update!(dye.tmp.sol₀, ts.RHS₁, ts.RHS₂, ts.RHS₃, ts.RHS₄, clock.dt)  
  FourierFlows.dealias!(ts.sol₀,grid);
  DyeVarsUpdates!(ts.sol₀, dye.ρ, prob.grid);
end

function RK4update!(sol, RHS₁, RHS₂, RHS₃, RHS₄, dt)
  @. sol += dt*(RHS₁ / 6 + RHS₂ / 3  + RHS₃ / 3 + RHS₄ / 6)
  
  return nothing
end

function RK4substeps!(sol, clock, ts,  vars, params, grid, t, dt)
  # Substep 1
  DyeEqn!(ts.RHS₁, sol, t, clock, vars, params, grid)
  
  # Substep 2
  substepsol!(ts.sol₁, sol, ts.RHS₁, dt/2)
  DyeEqn!(ts.RHS₂, ts.sol₁, t+dt/2, clock, vars, params, grid)
  
  # Substep 3
  substepsol!(ts.sol₁, sol, ts.RHS₂, dt/2)
  DyeEqn!(ts.RHS₃, ts.sol₁, t+dt/2, clock, vars, params, grid)
  
  # Substep 4
  substepsol!(ts.sol₁, sol, ts.RHS₃, dt)
  DyeEqn!(ts.RHS₄, ts.sol₁, t+dt, clock, vars, params, grid)
    
  return nothing
end

function substepsol!(newsol, sol, RHS, dt)
  @. newsol = sol + dt*RHS
  
  return nothing
end

function DyeVarsUpdates!(sol,ρ, grid)
	ldiv!(ρ, grid.rfftplan, deepcopy(sol))

	return nothing
end

function DyeEqn!(N, sol, t, clock, vars, params, grid)
    # δ function
    δ(a::Int,b::Int) = ( a == b ? 1 : 0 );
    
    # To emulate the dye tracing,  We define dye density as ρ, and conserve the continuity equation
    # Such that, ∂ρ/∂t = ∇ ⋅ (ρ \vec{v})
    # So that, in spectral space,
    # ∂ρₖ/∂t = ∑ᵢ -im*kᵢ*(ρₖ vₖᵢ)
    
    # Initialization of rho
    @. N*=0

    for (uᵢ,kᵢ) ∈ zip([vars.ux,vars.uy,vars.uz],[grid.kr,grid.l,grid.m])
         # Initialization
        @. vars.nonlin1 *= 0;
        ρuᵢ  = vars.nonlin1
        ρuᵢh = vars.nonlinh1

        # get back the updated sol in real space using fft
        ldiv!(ρuᵢ, grid.rfftplan, deepcopy(sol))  

        # Pre-Calculation in Real Space
        @. ρuᵢ *= uᵢ
        
        # Fourier transform 
        mul!(ρuᵢh, grid.rfftplan, ρuᵢ)
        
        # Perform the actual calculation
        @. N += -im*kᵢ*ρuᵢh
    end
    return nothing
end