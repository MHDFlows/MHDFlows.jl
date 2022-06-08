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
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum


# δ function
δ(a::Int,b::Int) = ( a == b ? 1 : 0 );


function UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")

  if direction == "x"

  	# a = {1,2,3} -> {x,y,z} direction
  	a    = 1;
  	kₐ   = grid.kr;
  	k⁻²  = grid.invKrsq;
  	uᵢh  = vars.uxh;
  	∂uᵢh∂t = @view N[:,:,:,params.ux_ind];

  elseif direction == "y"

  	a    = 2;
  	kₐ   = grid.l;
  	k⁻²  = grid.invKrsq;
  	uᵢh  = vars.uyh;
  	∂uᵢh∂t = @view N[:,:,:,params.uy_ind];
    
  elseif direction == "z"

  	a    = 3;
  	kₐ   = grid.m;
  	k⁻²  = grid.invKrsq;
  	uᵢh  = vars.uzh;
  	∂uᵢh∂t = @view N[:,:,:,params.uz_ind];

  else

  	@warn "Warning : Unknown direction is declerad"

  end

  @. ∂uᵢh∂t*= 0;

  for (uᵢ,kᵢ) ∈ zip([vars.ux,vars.uy,vars.uz],[grid.kr,grid.l,grid.m])
        for (uⱼ,kⱼ,j) ∈ zip([vars.ux,vars.uy,vars.uz],[grid.kr,grid.l,grid.m],[1, 2, 3])

          # Initialization 
          @. vars.nonlin1 *= 0;
          uᵢuⱼ  = vars.nonlin1;    
          uᵢuⱼh = vars.nonlinh1;
          
          # Pre-Calculation in Real Space
          @. uᵢuⱼ = uᵢ*uⱼ;

          # Fourier transform 
          mul!(uᵢuⱼh, grid.rfftplan, uᵢuⱼ);
          
          # Perform the actual calculation
          @. ∂uᵢh∂t += -im*kᵢ*(δ(a,j)-kₐ*kⱼ*k⁻²)*uᵢuⱼh;
            
        end
    end
    
    #Compute the diffusion term  - νk^2 u_i
    @. ∂uᵢh∂t += -grid.Krsq*params.ν*uᵢh;
    return nothing
    
end

function HDcalcN_advection!(N, sol, t, clock, vars, params, grid)

  #Update V  Fourier Conponment
  @. vars.uxh = sol[:, :, :, params.ux_ind];
  @. vars.uyh = sol[:, :, :, params.uy_ind];
  @. vars.uzh = sol[:, :, :, params.uz_ind];
    

  #Update V + B Real Conponment
  ldiv!(vars.ux, grid.rfftplan, deepcopy(vars.uxh))
  ldiv!(vars.uy, grid.rfftplan, deepcopy(vars.uyh))
  ldiv!(vars.uz, grid.rfftplan, deepcopy(vars.uzh))
  
  #Update V Advection
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y")
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z")
  
  return nothing
end

function HDupdatevars!(prob)
  vars, grid, sol, params = prob.vars, prob.grid, prob.sol, prob.params
  
  dealias!(sol, grid)
  
  #Update V Fourier Conponment
  @. vars.uxh = sol[:, :, :, params.ux_ind];
  @. vars.uyh = sol[:, :, :, params.uy_ind];
  @. vars.uzh = sol[:, :, :, params.uz_ind];


  #Update V + B Real Conponment
  ldiv!(vars.ux, grid.rfftplan, deepcopy(vars.uxh)) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.uy, grid.rfftplan, deepcopy(vars.uyh)) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.uz, grid.rfftplan, deepcopy(vars.uzh)) # deepcopy() since inverse real-fft destroys its input
 
  return nothing
end

end
