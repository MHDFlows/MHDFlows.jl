module HDSolver

# ----------
# Navier–Stokes Solver for 3D Hydrodynamics Problem
# ----------

export 
	UᵢUpdate!,
	HDcalcN_advection!,
	HDupdatevars!

using
  CUDA,
  TimerOutputs

using LinearAlgebra: mul!, ldiv!
include("VPSolver.jl")

# δ function
δ(a::Int,b::Int) = ( a == b ? 1 : 0 );

# checking function of VP method
VP_is_turned_on(params) = hasproperty(params,:U₀x);

function UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")

  if direction == "x"

  	# a = {1,2,3} -> {x,y,z} direction
  	a    = 1;
  	kₐ   = grid.kr;
  	k⁻²  = grid.invKrsq;
  	∂uᵢh∂t = @view N[:,:,:,params.ux_ind];

  elseif direction == "y"

  	a    = 2;
  	kₐ   = grid.l;
  	k⁻²  = grid.invKrsq;
  	∂uᵢh∂t = @view N[:,:,:,params.uy_ind];
    
  elseif direction == "z"

  	a    = 3;
  	kₐ   = grid.m;
  	k⁻²  = grid.invKrsq;
  	∂uᵢh∂t = @view N[:,:,:,params.uz_ind];

  else

  	error("Warning : Unknown direction is declerad")

  end

  @. ∂uᵢh∂t*= 0;
  uᵢuⱼ  = vars.nonlin1;    
  uᵢuⱼh = vars.nonlinh1;
  for (uᵢ,kᵢ) ∈ zip([vars.ux,vars.uy,vars.uz],[grid.kr,grid.l,grid.m])
    for (uⱼ,kⱼ,j) ∈ zip([vars.ux,vars.uy,vars.uz],[grid.kr,grid.l,grid.m],[1, 2, 3])
      
      @timeit_debug params.debugTimer "Pseudo" begin
        # Pre-Calculation in Real Space
        @. uᵢuⱼ = uᵢ*uⱼ;
      end

      @timeit_debug params.debugTimer "Spectral" begin
        # Fourier transform 
        mul!(uᵢuⱼh, grid.rfftplan, uᵢuⱼ);
      end

      @timeit_debug params.debugTimer "Advection" begin
        # Perform the actual calculation
        @. ∂uᵢh∂t += -im*kᵢ*(δ(a,j)-kₐ*kⱼ*k⁻²)*uᵢuⱼh;
      end

    end
  end
  
  # Updating the solid domain if VP flag is ON
  if VP_is_turned_on(params) 
    @timeit_debug params.debugTimer "VP Uᵢ" VPSolver.VP_UᵢUpdate!(∂uᵢh∂t, kₐ.*k⁻², a, clock, vars, params, grid)
  end

  #Compute the diffusion term  - νk^2 u_i
  uᵢ = direction == "x" ? vars.ux : direction == "y" ? vars.uy : vars.uz;
  uᵢh = vars.nonlinh1;
  mul!(uᵢh, grid.rfftplan, uᵢ); 
  @. ∂uᵢh∂t += -grid.Krsq*params.ν*uᵢh;

  # hyperdiffusion term
  if params.nν > 1
    @. ∂uᵢh∂t += -grid.Krsq^params.nν*params.ν*uᵢh;
  end

  return nothing
    
end

function HDcalcN_advection!(N, sol, t, clock, vars, params, grid)

  #Update V + B Real Conponment
  ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]));
  ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]));
  ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]));
  
  #Update V Advection
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y")
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z")
  
  return nothing
end

end
