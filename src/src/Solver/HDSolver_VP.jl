module HDSolver_VP
# ----------
# Navier–Stokes Solver for 3D Hydrodynamics Problem with Volume Penalization Method
# ----------
export 
	UᵢUpdate!,
	HDcalcN_advection!

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

function HDcalcN_advection!(N, sol, t, clock, vars, params, grid)

  #Update V + B Fourier Conponment
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

end