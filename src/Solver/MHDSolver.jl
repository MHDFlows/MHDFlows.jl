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
  TimerOutputs

using LinearAlgebra: mul!, ldiv!
include("VPSolver.jl")

# δ notation
δ(a::Int,b::Int) = ( a == b ? 1 : 0 )
# ϵ notation
ϵ(i::Int,j::Int,k::Int) = (i - j)*(j - k)*(k - i)/2

# checking function of VP method
VP_is_turned_on(params) = hasproperty(params,:U₀x);

function UᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="x")

  if direction == "x"

  	# a = {1,2,3} -> {x,y,z} direction
  	a    = 1
  	kₐ   = grid.kr
  	k⁻²  = grid.invKrsq
  	∂uᵢh∂t = @view N[:,:,:,params.ux_ind]

  elseif direction == "y"

  	a    = 2
  	kₐ   = grid.l
  	k⁻²  = grid.invKrsq
  	∂uᵢh∂t = @view N[:,:,:,params.uy_ind]

  elseif direction == "z"
  	a    = 3
  	kₐ   = grid.m
  	k⁻²  = grid.invKrsq
  	∂uᵢh∂t = @view N[:,:,:,params.uz_ind]

  else

  	error("Warning : Unknown direction is declerad")

  end
  #idea : we are computing ∂uᵢh∂t = im*kᵢ*(δₐⱼ - kₐkⱼk⁻²)*(bᵢbⱼ - uᵢuⱼh) 
  #  as uᵢuⱼ = uⱼuᵢ in our case
  #     1  2  3
  #   1 11 12 13
  #   2 21 22 23 , part of computation is repeated, 11(1),12(2),13(2),22(1),23(2),33(1)
  #   3 31 32 33
  #   Their only difference for u_ij is the advection part
  @. ∂uᵢh∂t*= 0;
  for (bᵢ,uᵢ,kᵢ,i) ∈ zip((vars.bx,vars.by,vars.bz),(vars.ux,vars.uy,vars.uz),(grid.kr,grid.l,grid.m),(1,2,3))
    for (bⱼ,uⱼ,kⱼ,j) ∈ zip((vars.bx,vars.by,vars.bz),(vars.ux,vars.uy,vars.uz),(grid.kr,grid.l,grid.m),(1, 2, 3))
      if j >= i
        # Initialization
        @. vars.nonlin1  *= 0
        @. vars.nonlinh1 *= 0
        bᵢbⱼ_minus_uᵢuⱼ  = vars.nonlin1  
        bᵢbⱼ_minus_uᵢuⱼh = vars.nonlinh1

        # Perform Computation in Real space
        @. bᵢbⱼ_minus_uᵢuⱼ = bᵢ*bⱼ - uᵢ*uⱼ
        mul!(bᵢbⱼ_minus_uᵢuⱼh, grid.rfftplan, bᵢbⱼ_minus_uᵢuⱼ)

        # Perform the Actual Advection update
        @. ∂uᵢh∂t += im*kᵢ*(δ(a,j)-kₐ*kⱼ*k⁻²)*bᵢbⱼ_minus_uᵢuⱼh
        if i != j  # repeat the calculation for u_ij
          @. ∂uᵢh∂t += im*kⱼ*(δ(a,i)-kₐ*kᵢ*k⁻²)*bᵢbⱼ_minus_uᵢuⱼh
        end
      end
    end
  end

  # Updating the solid domain if VP flag is ON
  if VP_is_turned_on(params) 
    VPSolver.VP_UᵢUpdate!(∂uᵢh∂t, kₐ.*k⁻², a, clock, vars, params, grid)
  end

  #Compute the diffusion term  - νk^2 u_i
  uᵢ = direction == "x" ? vars.ux : direction == "y" ? vars.uy : vars.uz;
  uᵢh = vars.nonlinh1
  mul!(uᵢh, grid.rfftplan, uᵢ)
  @. ∂uᵢh∂t += -grid.Krsq*params.ν*uᵢh
  
  # hyperdiffusion term
  if params.nν > 1
    @. ∂uᵢh∂t += -grid.Krsq^params.nν*params.ν*uᵢh
  end

  return nothing
    
end

# B function
function BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")

	#To Update B_i, we have two terms to compute:
	# ∂B_i/∂t = im ∑_j k_j*(b_iu_j - u_ib_j)  - η k^2 B_i
	#We split it into two part for sparating the computation.

  # declare the var u_i, b_i for computation
	if direction == "x"
    a   = 1;
    kₐ  = grid.kr;
    k⁻² = grid.invKrsq;
		uᵢ  = vars.ux;
		bᵢ  = vars.bx; 
		∂Bᵢh∂t = @view N[:,:,:,params.bx_ind];

	elseif direction == "y"
    a   = 2;
    kₐ  = grid.l;
    k⁻² = grid.invKrsq;
		uᵢ  = vars.uy;
		bᵢ  = vars.by; 
		∂Bᵢh∂t = @view N[:,:,:,params.by_ind];

	elseif direction == "z"
    a   = 3;
    kₐ  = grid.m;
    k⁻² = grid.invKrsq;
		uᵢ  = vars.uz;
		bᵢ  = vars.bz; 
		∂Bᵢh∂t = @view N[:,:,:,params.bz_ind];

	else

		@warn "Warning : Unknown direction is declerad"

	end

  @. ∂Bᵢh∂t*= 0;
  uᵢbⱼ_minus_bᵢuⱼ  = vars.nonlin1;        
  uᵢbⱼ_minus_bᵢuⱼh = vars.nonlinh1;
  #Compute the first term, im ∑_j k_j*(b_iu_j - u_ib_j)
  for (bⱼ,uⱼ,kⱼ,j) ∈ zip((vars.bx,vars.by,vars.bz),(vars.ux,vars.uy,vars.uz),(grid.kr,grid.l,grid.m),(1,2,3))
    if a != j
      # Perform Computation in Real space
      @. uᵢbⱼ_minus_bᵢuⱼ = uᵢ*bⱼ - bᵢ*uⱼ;
      
      mul!(uᵢbⱼ_minus_bᵢuⱼh, grid.rfftplan, uᵢbⱼ_minus_bᵢuⱼ);

      # Perform the Actual Advection update
      @. ∂Bᵢh∂t += im*kⱼ*uᵢbⱼ_minus_bᵢuⱼh;  

    end
  end

  # Updating the solid domain if VP flag is ON
  if VP_is_turned_on(params) 
    VPSolver.VP_BᵢUpdate!(∂Bᵢh∂t, kₐ.*k⁻², a, clock, vars, params, grid)
  end

  #Compute the diffusion term  - ηk^2 B_i
  bᵢh = vars.nonlinh1;
  mul!(bᵢh, grid.rfftplan, bᵢ); 
  @. ∂Bᵢh∂t += -grid.Krsq*params.η*bᵢh;

  # hyperdiffusion term
  if params.nη > 1
    @. ∂Bᵢh∂t += -grid.Krsq^params.nη*params.η*bᵢh;
  end
    
    return nothing

end

# B function for EMHD system
# For E-MHD system, the induction will be changed into
#  ∂B/∂t = -dᵢ * ∇× [ (∇× B) × B ] + η ∇²B
# In this function, we will implement the equation and assume dᵢ = 1
function EMHD_BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")

  # To Update B_i, we have to first break down the equation :
  # ∂B/∂t  = - ∇× [ (∇× B) × B ] + η ∇²B
  # Let A = (∇× B). By using vector calculus identities, we have
  # ∂B/∂t  = - [ (∇ ⋅ B + B ⋅ ∇)A - (∇ ⋅ A + A ⋅ ∇)B ]  + η ∇²B
  # Using ∇ ⋅ B  = 0 and vector calculus identities ∇⋅(∇× B) = 0, we finally get the expression
  # ∂B/∂t  = - [(B ⋅ ∇)A - (A ⋅ ∇)B ]  + η ∇²B =  (A ⋅ ∇)B - (B ⋅ ∇)A  + η ∇²B
  # For any direction i, we will have the following expression in k-space
  # 𝔉(∂Bᵢ/∂t)  = 𝔉[(Aⱼ∂ⱼ)Bᵢ - Bⱼ∂ⱼAᵢ] -  k²𝔉(B)
  # To compute the first term in RHS, we break it into three step
  # 1. compute real space term ∂ⱼBᵢ using spectral method
  # 2. compute Aⱼ∂ⱼBᵢ using pseudo spectral method
  # 3. add the answer to 𝔉(∂Bᵢ/∂t) 
  #

  # declare the var u_i, b_i for computation
  if direction == "x"
    a   = 1
    kₐ  = grid.kr
    Aᵢ  = vars.∇XBᵢ
    bᵢ  = vars.bx 
    bᵢh = @view sol[:,:,:,params.bx_ind]
    ∂Bᵢh∂t = @view N[:,:,:,params.bx_ind]

  elseif direction == "y"
    a   = 2
    kₐ  = grid.l
    Aᵢ  = vars.∇XBⱼ
    bᵢ  = vars.by 
    bᵢh = @view sol[:,:,:,params.by_ind]
    ∂Bᵢh∂t = @view N[:,:,:,params.by_ind]

  elseif direction == "z"
    a   = 3
    kₐ  = grid.m
    Aᵢ  = vars.∇XBₖ
    bᵢ  = vars.bz 
    bᵢh = @view sol[:,:,:,params.bz_ind]
    ∂Bᵢh∂t = @view N[:,:,:,params.bz_ind]
  else

    @warn "Warning : Unknown direction is declerad"

  end

  A₁  = vars.∇XBᵢ
  A₂  = vars.∇XBⱼ
  A₃  = vars.∇XBₖ

  # define the sketch array
  ∂ⱼAᵢ  = ∂ⱼBᵢ  = vars.nonlin1
  Bⱼ∂ⱼAᵢ= Aⱼ∂ⱼBᵢ= vars.nonlin1
  Aᵢh   = Bᵢh   = vars.nonlinh1
  ∂ⱼAᵢh = ∂ⱼBᵢh = vars.nonlinh1
  Bⱼ∂ⱼAᵢh = Aⱼ∂ⱼBᵢh = vars.nonlinh1

  @. ∂Bᵢh∂t*= 0;
  for (bⱼ,Aⱼ,kⱼ) ∈ zip((vars.bx,vars.by,vars.bz),(A₁,A₂,A₃),(grid.kr,grid.l,grid.m))
    
    # first step
    @. Aᵢh = 0
    mul!(Aᵢh, grid.rfftplan, Aᵢ)
    @. ∂ⱼAᵢh = im*kⱼ*Aᵢh
    ldiv!(∂ⱼAᵢ, grid.rfftplan, deepcopy(∂ⱼAᵢh))
    # second step
    @. Bⱼ∂ⱼAᵢ = bⱼ*∂ⱼAᵢ
    @. Bⱼ∂ⱼAᵢh = 0
    mul!(Bⱼ∂ⱼAᵢh, grid.rfftplan, Bⱼ∂ⱼAᵢ)
    # final step
    @. ∂Bᵢh∂t -= Bⱼ∂ⱼAᵢh

    # first step
    @. ∂ⱼBᵢ = 0
    @. ∂ⱼBᵢh = im*kⱼ*bᵢh
    ldiv!(∂ⱼBᵢ, grid.rfftplan, deepcopy(∂ⱼBᵢh))
    # second step
    @. Aⱼ∂ⱼBᵢ = Aⱼ*∂ⱼBᵢ
    @. Aⱼ∂ⱼBᵢh = 0
    mul!(Aⱼ∂ⱼBᵢh, grid.rfftplan, Aⱼ∂ⱼBᵢ)
    # final step
    @. ∂Bᵢh∂t += Aⱼ∂ⱼBᵢh

  end

  return nothing
  
end

# Compute the ∇XB term
function Get∇XB!(sol, vars, params, grid)

  # ∇XB = im*( k × B )ₖ = im*ϵ_ijk kᵢ Bⱼ

  # define the variables
  k₁,k₂,k₃ = grid.kr,grid.l,grid.m;
  B₁h = @view sol[:,:,:,params.bx_ind]
  B₂h = @view sol[:,:,:,params.by_ind]
  B₃h = @view sol[:,:,:,params.bz_ind]
  A₁  = vars.∇XBᵢ
  A₂  = vars.∇XBⱼ
  A₃  = vars.∇XBₖ

  # Way 1  of appling Curl
  #=∇XBₖh = vars.nonlinh1
  for (∇XBₖ ,k) ∈ zip((A₁,A₂,A₃),(1,2,3))
    @. ∇XBₖh*=0
    for (Bⱼh,j)  ∈ zip((B₁h,B₂h,B₃h),(1,2,3))
      for (kᵢ,i)  ∈ zip((k₁,k₂,k₃),(1,2,3))
        if ϵ(i,j,k) != 0
          @. ∇XBₖh += im*ϵ(i,j,k)*kᵢ*Bⱼh
        end
      end
    end
    ldiv!(∇XBₖ, grid.rfftplan, deepcopy( ∇XBₖh))
  end=#

  # Way 2  of appling Curl
  CBᵢh = vars.nonlinh1
  @. CBᵢh = im*(k₂*B₃h - k₃*B₂h)
  ldiv!(A₁, grid.rfftplan, CBᵢh)  

  @. CBᵢh = im*(k₃*B₁h - k₁*B₃h)
  ldiv!(A₂, grid.rfftplan, CBᵢh)  

  @. CBᵢh = im*(k₁*B₂h - k₂*B₁h)
  ldiv!(A₃, grid.rfftplan, CBᵢh)  

  return nothing
end

function EMHDcalcN_advection!(N, sol, t, clock, vars, params, grid)

  #Update B Advection
  Get∇XB!(sol, vars, params, grid)
  EMHD_BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")
  EMHD_BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y")
  EMHD_BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z")

  #Update B Real Conponment
  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]))
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]))
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind]))

  return nothing
end

function MHDcalcN_advection!(N, sol, t, clock, vars, params, grid)

  #Update V + B Real Conponment
  ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]));
  ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]));
  ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]));
  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]));
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]));
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])); 

  #Update V Advection
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x");
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y");
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z");

  #Update B Advection
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x");
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y");
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z"); 

  return nothing
end

end
