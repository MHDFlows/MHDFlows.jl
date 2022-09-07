module MHDSolver
# ----------
# Navier–Stokes Solver for 3D Magnetohydrodynamics Problem
# ----------

export 
	UᵢUpdate!,
	BᵢUpdate!,
	MHDcalcN_advection!,
	MHDupdatevars!


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

  for (bᵢ,uᵢ,kᵢ) ∈ zip([vars.bx,vars.by,vars.bz],[vars.ux,vars.uy,vars.uz],[grid.kr,grid.l,grid.m])
        for (bⱼ,uⱼ,kⱼ,j) ∈ zip([vars.bx,vars.by,vars.bz],[vars.ux,vars.uy,vars.uz],[grid.kr,grid.l,grid.m],[1, 2, 3])
          
          # Initialization
          @. vars.nonlin1  *= 0;
          @. vars.nonlinh1 *= 0;
          bᵢbⱼ_minus_uᵢuⱼ  = vars.nonlin1;  
          bᵢbⱼ_minus_uᵢuⱼh = vars.nonlinh1;
          # Perform Computation in Real space
          @. bᵢbⱼ_minus_uᵢuⱼ = bᵢ*bⱼ - uᵢ*uⱼ;
          mul!(bᵢbⱼ_minus_uᵢuⱼh, grid.rfftplan, bᵢbⱼ_minus_uᵢuⱼ);

          # Perform the Actual Advection update
          @. ∂uᵢh∂t += im*kᵢ*(δ(a,j)-kₐ*kⱼ*k⁻²)*bᵢbⱼ_minus_uᵢuⱼh;
            
        end
    end
    
    #Compute the diffusion term  - νk^2 u_i
    @. ∂uᵢh∂t += -grid.Krsq*params.ν*uᵢh;
    
    #=# hyperdiffusion term
    if params.nν > 1
      @. ∂uᵢh∂t += -grid.Krsq^params.nν*params.ν*uᵢh;
    end=#

    return nothing
    
end

# B function
function BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")

	#To Update B_i, we have two terms to compute:
	# ∂B_i/∂t = im ∑_j k_j*(b_iu_j - u_ib_j)  - η k^2 B_i
	#We split it into two part for sparating the computation.

  # declare the var u_i, b_i for computation
	if direction == "x"

		uᵢ  = vars.ux;
		bᵢ  = vars.bx; 
		bᵢh = vars.bxh; 
		∂Bᵢh∂t = @view N[:,:,:,params.bx_ind];

	elseif direction == "y"

		uᵢ  = vars.uy;
		bᵢ  = vars.by; 
		bᵢh = vars.byh; 
		∂Bᵢh∂t = @view N[:,:,:,params.by_ind];

	elseif direction == "z"

		uᵢ  = vars.uz;
		bᵢ  = vars.bz; 
		bᵢh = vars.bzh; 
		∂Bᵢh∂t = @view N[:,:,:,params.bz_ind];

	else

		@warn "Warning : Unknown direction is declerad"

	end

    @. ∂Bᵢh∂t*= 0;
    
    #Compute the first term, im ∑_j k_j*(b_iu_j - u_ib_j)
    for (bⱼ,uⱼ,kⱼ) ∈ zip([vars.bx,vars.by,vars.bz],[vars.ux,vars.uy,vars.uz],[grid.kr,grid.l,grid.m])

        # Initialization 
        @. vars.nonlin1  *= 0;
        @. vars.nonlinh1 *= 0;
        uᵢbⱼ_minus_bᵢuⱼ  = vars.nonlin1;        
        uᵢbⱼ_minus_bᵢuⱼh = vars.nonlinh1;
        # Perform Computation in Real space
        @. uᵢbⱼ_minus_bᵢuⱼ = uᵢ*bⱼ - bᵢ*uⱼ;
        mul!(uᵢbⱼ_minus_bᵢuⱼh, grid.rfftplan, uᵢbⱼ_minus_bᵢuⱼ);
        # Perform the Actual Advection update
        @. ∂Bᵢh∂t += im*kⱼ*uᵢbⱼ_minus_bᵢuⱼh;  

    end
    
    #Compute the diffusion term  - ηk^2 B_i
    @. ∂Bᵢh∂t += -grid.Krsq*params.η*bᵢh;

    # hyperdiffusion term
    if params.nη > 1
      @. ∂Bᵢh∂t += -grid.Krsq^params.nη*params.η*bᵢh;
    end
    
    return nothing

end

function MHDcalcN_advection!(N, sol, t, clock, vars, params, grid)

  #Update V + B Fourier Conponment
  copyto!(vars.uxh, @view sol[:, :, :, params.ux_ind]);
  copyto!(vars.uyh, @view sol[:, :, :, params.uy_ind]);
  copyto!(vars.uzh, @view sol[:, :, :, params.uz_ind]);
    
  copyto!(vars.bxh, @view sol[:, :, :, params.bx_ind]);
  copyto!(vars.byh, @view sol[:, :, :, params.by_ind]);
  copyto!(vars.bzh, @view sol[:, :, :, params.bz_ind]);

  #Update V + B Real Conponment
  ldiv!(vars.ux, grid.rfftplan, deepcopy(vars.uxh))
  ldiv!(vars.uy, grid.rfftplan, deepcopy(vars.uyh))
  ldiv!(vars.uz, grid.rfftplan, deepcopy(vars.uzh))
  ldiv!(vars.bx, grid.rfftplan, deepcopy(vars.bxh))
  ldiv!(vars.by, grid.rfftplan, deepcopy(vars.byh))
  ldiv!(vars.bz, grid.rfftplan, deepcopy(vars.bzh))  
  
  #Update V Advection
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y")
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z")
  
  #Update B Advection
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y")
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z")  
  
  return nothing
end

function MHDupdatevars!(prob)
  vars, grid, sol, params = prob.vars, prob.grid, prob.sol, prob.params
  
  dealias!(sol, grid)
  
  #Update V + B Fourier Conponment
  copyto!(vars.uxh, @view sol[:, :, :, params.ux_ind]);
  copyto!(vars.uyh, @view sol[:, :, :, params.uy_ind]);
  copyto!(vars.uzh, @view sol[:, :, :, params.uz_ind]);

  copyto!(vars.bxh, @view sol[:, :, :, params.bx_ind]);
  copyto!(vars.byh, @view sol[:, :, :, params.by_ind]);
  copyto!(vars.bzh, @view sol[:, :, :, params.bz_ind]);

  #Update V + B Real Conponment
  ldiv!(vars.ux, grid.rfftplan, deepcopy(vars.uxh)) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.uy, grid.rfftplan, deepcopy(vars.uyh)) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.uz, grid.rfftplan, deepcopy(vars.uzh)) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.bx, grid.rfftplan, deepcopy(vars.bxh)) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.by, grid.rfftplan, deepcopy(vars.byh)) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.bz, grid.rfftplan, deepcopy(vars.bzh)) # deepcopy() since inverse real-fft destroys its input
  
  return nothing
end

end
