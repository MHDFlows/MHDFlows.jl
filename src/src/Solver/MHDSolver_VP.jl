module MHDSolver_VP
# ----------
# Navier–Stokes Solver for 3D Magnetohydrodynamics Problem with Volume Penalization Method
# ----------
export 
	UᵢUpdate!,
	BᵢUpdate!,
	MHDcalcN_advection!,
  DivBCorrection!,
  DivVCorrection!

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

  T = eltype(grid);
  
  if direction == "x"

  	# a = {1,2,3} -> {x,y,z} direction
  	a    = 1;
  	kₐ   = grid.kr;
  	k⁻²  = grid.invKrsq;
    U₀   = params.U₀x;
    uᵢ   = vars.ux;
  	uᵢh  = vars.uxh;
  	∂uᵢh∂t = @view N[:,:,:,params.ux_ind];

  elseif direction == "y"

  	a    = 2;
  	kₐ   = grid.l;
  	k⁻²  = grid.invKrsq;
    U₀   = params.U₀y;
    uᵢ   = vars.uy;
  	uᵢh  = vars.uyh;
  	∂uᵢh∂t = @view N[:,:,:,params.uy_ind];

  elseif direction == "z"

  	a    = 3;
  	kₐ   = grid.m;
  	k⁻²  = grid.invKrsq;
    U₀   = params.U₀z;
    uᵢ   = vars.uz;
  	uᵢh  = vars.uzh;
  	∂uᵢh∂t = @view N[:,:,:,params.uz_ind];

  else

  	@warn "Warning : Unknown direction is declerad"

  end

  @. ∂uᵢh∂t*= 0;
  
  η = clock.dt*13/7; #η condition for AB3 Method
  χ = params.χ;  
  for (bᵢ,uᵢ,kᵢ) ∈ zip([vars.bx,vars.by,vars.bz],[vars.ux,vars.uy,vars.uz],[grid.kr,grid.l,grid.m])
        for (bⱼ,uⱼ,kⱼ,j) ∈ zip([vars.bx,vars.by,vars.bz],[vars.ux,vars.uy,vars.uz],[grid.kr,grid.l,grid.m],[1, 2, 3])

          @. vars.nonlin1  *= 0;
          @. vars.nonlinh1 *= 0;
          bᵢbⱼ_minus_uᵢuⱼ  = vars.nonlin1;  
          bᵢbⱼ_minus_uᵢuⱼh = vars.nonlinh1;
          @. bᵢbⱼ_minus_uᵢuⱼ = bᵢ*bⱼ - uᵢ*uⱼ;
          mul!(bᵢbⱼ_minus_uᵢuⱼh, grid.rfftplan, bᵢbⱼ_minus_uᵢuⱼ);

          # Perform the Actual Advection update
          @. ∂uᵢh∂t += im*kᵢ*(δ(a,j)-kₐ*kⱼ*k⁻²)*bᵢbⱼ_minus_uᵢuⱼh;
            
        end
    end
    
    for (uⱼ,Uⱼ,kⱼ,j) ∈ zip([vars.ux,vars.uy,vars.uz],[params.U₀x,params.U₀y,params.U₀z],[grid.kr,grid.l,grid.m],[1, 2, 3])

        #The Volume Penalization term, Assuming U_wall = Uⱼ , j ∈ [x,y,z] direction
        @. vars.nonlin1  *= 0;
        @. vars.nonlinh1 *= 0;  
        χUᵢ_η  = vars.nonlin1; 
        χUᵢ_ηh = vars.nonlinh1;
        @. χUᵢ_η  =  χ/η*(uⱼ - Uⱼ);
        mul!(χUᵢ_ηh, grid.rfftplan,  χUᵢ_η);  
          
        # Perform the Actual Advection update
        @. ∂uᵢh∂t += -(δ(a,j)-kₐ*kⱼ*k⁻²)*χUᵢ_ηh;  
    end

    #Compute the diffusion term  - νk^2 u_i
    @. ∂uᵢh∂t += -grid.Krsq*params.ν*uᵢh;

    # hyperdiffusion term
    if params.nν > 1
      @. ∂uᵢh∂t += -grid.Krsq^params.nν*params.ν*uᵢh;
    end

    return nothing
    
end

function BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")

	#To Update B_i, we have two terms to compute:
	# ∂B_i/∂t = im ∑_j k_j*(b_iu_j - u_ib_j)  - η k^2 B_i
	#We split it into two part for sparating the computation.

  # declare the var u_i, b_i for computation
	if direction == "x"
    a   = 1;
    kₐ  = grid.kr;
    k⁻² = grid.invKrsq;
    B₀  = params.B₀x;
		uᵢ  = vars.ux;
		bᵢ  = vars.bx; 
		bᵢh = vars.bxh; 
		∂Bᵢh∂t = @view N[:,:,:,params.bx_ind];

	elseif direction == "y"
    a   = 2;
    kₐ  = grid.l;
    k⁻² = grid.invKrsq;
    B₀  = params.B₀y;
		uᵢ  = vars.uy;
		bᵢ  = vars.by; 
		bᵢh = vars.byh; 
		∂Bᵢh∂t = @view N[:,:,:,params.by_ind];

	elseif direction == "z"
    a   = 3;
    kₐ  = grid.m;
    k⁻² = grid.invKrsq;
    B₀  = params.B₀z;
		uᵢ  = vars.uz;
		bᵢ  = vars.bz; 
		bᵢh = vars.bzh; 
		∂Bᵢh∂t = @view N[:,:,:,params.bz_ind];

	else

		@warn "Warning : Unknown direction is declerad"

	end

    @. ∂Bᵢh∂t*= 0;
    
    η = clock.dt*13/7; #η condition for AB3 Method
    χ = params.χ;  

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

    for (bⱼ,Bⱼ,kⱼ,j) ∈ zip([vars.bx,vars.by,vars.bz],[params.B₀x,params.B₀y,params.B₀z],[grid.kr,grid.l,grid.m],[1, 2, 3])
      #The Volume Penalization term, Assuming B_wall = Bⱼ, j ∈ [x,y,z] direction
      @. vars.nonlin1  *= 0;
      @. vars.nonlinh1 *= 0;  
      χbᵢ_η  = vars.nonlin1;
      χbᵢ_ηh = vars.nonlinh1;
      @.  χbᵢ_η  = χ/η*(bⱼ - Bⱼ);
      mul!(χbᵢ_ηh, grid.rfftplan, χbᵢ_η);

      # Perform the Actual Advection update
      @. ∂Bᵢh∂t += -(δ(a,j)-kₐ*kⱼ*k⁻²)*χbᵢ_ηh;
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
  @. vars.uxh = sol[:, :, :, params.ux_ind];
  @. vars.uyh = sol[:, :, :, params.uy_ind];
  @. vars.uzh = sol[:, :, :, params.uz_ind];
    
  @. vars.bxh = sol[:, :, :, params.bx_ind];
  @. vars.byh = sol[:, :, :, params.by_ind];
  @. vars.bzh = sol[:, :, :, params.bz_ind];

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

function DivBCorrection!(prob)
#= 
   Possion Solver for periodic boundary condition
   As in VP method, ∇ ⋅ B = 0 doesn't hold, B_{t+1} = ∇×Ψ + ∇Φ -> ∇ ⋅ B = ∇² Φ
   We need to find Φ and remove it using a Poission Solver 
   Here we are using the Fourier Method to find the Φ
   In Real Space,  
   ∇² Φ = ∇ ⋅ B   
   In k-Space,  
   ∑ᵢ -(kᵢ)² Φₖ = i∑ᵢ kᵢ(Bₖ)ᵢ
   Φ = F{ i∑ᵢ kᵢ (Bₖ)ᵢ / ∑ᵢ (k²)ᵢ}
=#  

  vars = prob.vars;
  grid = prob.grid;
  params = prob.params;
  #find Φₖ
  kᵢ,kⱼ,kₖ = grid.kr,grid.l,grid.m;
  k⁻² = grid.invKrsq;
  @. vars.nonlin1  *= 0;
  @. vars.nonlinh1 *= 0;       
  ∑ᵢkᵢBᵢh_k² = vars.nonlinh1;
  ∑ᵢkᵢBᵢ_k²  = vars.nonlin1;
  bxh = prob.sol[:, :, :, params.bx_ind];
  byh = prob.sol[:, :, :, params.by_ind];
  bzh = prob.sol[:, :, :, params.bz_ind];
  ∑ᵢkᵢBᵢh_k² = @. -im*(kᵢ*bxh + kⱼ*byh + kₖ*bzh);
  ∑ᵢkᵢBᵢh_k² = @. ∑ᵢkᵢBᵢh_k²*k⁻²;  # Φₖ
  
  # B  = B* - ∇Φ = Bᵢ - kᵢΦₖ  
  bxh  .-= kᵢ.*∑ᵢkᵢBᵢh_k²;
  byh  .-= kⱼ.*∑ᵢkᵢBᵢh_k²;
  bzh  .-= kₖ.*∑ᵢkᵢBᵢh_k²;
  
  #Update to Real Space vars
  ldiv!(vars.bx, grid.rfftplan, deepcopy(bxh));# deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.by, grid.rfftplan, deepcopy(byh));# deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.bz, grid.rfftplan, deepcopy(bzh));# deepcopy() since inverse real-fft destroys its input
end

function DivVCorrection!(prob)
#= 
   Possion Solver for periodic boundary condition
   As in VP method, ∇ ⋅ B = 0 doesn't hold, B_{t+1} = ∇×Ψ + ∇Φ -> ∇ ⋅ B = ∇² Φ
   We need to find Φ and remove it using a Poission Solver 
   Here we are using the Fourier Method to find the Φ
   In Real Space,  
   ∇² Φ = ∇ ⋅ B   
   In k-Space,  
   ∑ᵢ -(kᵢ)² Φₖ = i∑ᵢ kᵢ(Bₖ)ᵢ
   Φ = F{ i∑ᵢ kᵢ (Bₖ)ᵢ / ∑ᵢ (k²)ᵢ}
=#  

  vars = prob.vars;
  grid = prob.grid;
  params = prob.params;
  #find Φₖ
  kᵢ,kⱼ,kₖ = grid.kr,grid.l,grid.m;
  k⁻² = grid.invKrsq;
  @. vars.nonlin1  *= 0;
  @. vars.nonlinh1 *= 0;       
  ∑ᵢkᵢUᵢh_k² = vars.nonlinh1;
  ∑ᵢkᵢUᵢ_k²  = vars.nonlin1;
  uxh = prob.sol[:, :, :, params.ux_ind];
  uyh = prob.sol[:, :, :, params.uy_ind];
  uzh = prob.sol[:, :, :, params.uz_ind];
  ∑ᵢkᵢUᵢh_k² = @. -im*(kᵢ*uxh + kⱼ*uyh + kₖ*uzh);
  ∑ᵢkᵢUᵢh_k² = @. ∑ᵢkᵢUᵢh_k²*k⁻²;  # Φₖ
  
  # B  = B* - ∇Φ = Bᵢ - kᵢΦₖ  
  uxh  .-= kᵢ.*∑ᵢkᵢUᵢh_k²;
  uyh  .-= kⱼ.*∑ᵢkᵢUᵢh_k²;
  uzh  .-= kₖ.*∑ᵢkᵢUᵢh_k²;
  
  #Update to Real Space vars
  ldiv!(vars.ux, grid.rfftplan, deepcopy(uxh));# deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.uy, grid.rfftplan, deepcopy(uyh));# deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.uz, grid.rfftplan, deepcopy(uzh));# deepcopy() since inverse real-fft destroys its input
end

end