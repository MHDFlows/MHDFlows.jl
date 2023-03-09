module VPSolver

# ----------
# Volume Penalization Solver for HD/MHD N.S. equation
# ----------
export
  VP_BᵢUpdate,
  VP_UᵢUpdate!,
  DivBCorrection!,
  DivVCorrection!

using
  CUDA,
  TimerOutputs

using LinearAlgebra: mul!, ldiv!

# δ function
δ(a::Int,b::Int) = ( a == b ? 1 : 0 );

function VP_UᵢUpdate!(∂uᵢh∂t, kₐk⁻², a::Int, clock, vars, params, grid)
  χ = params.χ; 
  η = clock.dt*13/7; #η condition for AB3 Method
  for (uⱼ,Uⱼ,kⱼ,j) ∈ zip((vars.ux,vars.uy,vars.uz),(params.U₀x,params.U₀y,params.U₀z),(grid.kr,grid.l,grid.m),(1, 2, 3))
    
    #The Volume Penalization term, Assuming U_wall = Uⱼ , j ∈ [x,y,z] direction
    χUᵢ_η  = vars.nonlin1 
    χUᵢ_ηh = vars.nonlinh1
    @. χUᵢ_η  =  χ/η*(uⱼ - Uⱼ)

    mul!(χUᵢ_ηh, grid.rfftplan,  χUᵢ_η)
  
    # Perform the Actual Advection update
    @. ∂uᵢh∂t += -(δ(a,j)-kⱼ*kₐk⁻²)*χUᵢ_ηh 

  end
  return nothing;
end

function VP_BᵢUpdate!(∂Bᵢh∂t, kₐk⁻², a::Int, clock, vars, params, grid)

  χ = params.χ
  η = clock.dt*13/7 #η condition for AB3 Method
  
  for (bⱼ,Bⱼ,kⱼ,j) ∈ zip((vars.bx,vars.by,vars.bz),(params.B₀x,params.B₀y,params.B₀z),(grid.kr,grid.l,grid.m),(1, 2, 3))
    
    #The Volume Penalization term, Assuming B_wall = Bⱼ, j ∈ [x,y,z] direction
    χbᵢ_η  = vars.nonlin1
    χbᵢ_ηh = vars.nonlinh1
    @.  χbᵢ_η  = χ/η*(bⱼ - Bⱼ)

    mul!(χbᵢ_ηh, grid.rfftplan, χbᵢ_η)

    # Perform the Actual Advection update
    @. ∂Bᵢh∂t += -(δ(a,j)-kⱼ*kₐk⁻²)*χbᵢ_ηh

  end
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

  vars = prob.vars
  grid = prob.grid
  params = prob.params
  #find Φₖ
  kᵢ,kⱼ,kₖ = grid.kr,grid.l,grid.m
  k⁻² = grid.invKrsq
  @. vars.nonlin1  *= 0
  @. vars.nonlinh1 *= 0       
  ∑ᵢkᵢBᵢh_k² = vars.nonlinh1
  ∑ᵢkᵢBᵢ_k²  = vars.nonlin1
  @views bxh = prob.sol[:, :, :, params.bx_ind]
  @views byh = prob.sol[:, :, :, params.by_ind]
  @views bzh = prob.sol[:, :, :, params.bz_ind]
  @. ∑ᵢkᵢBᵢh_k² = -im*(kᵢ*bxh + kⱼ*byh + kₖ*bzh)
  @. ∑ᵢkᵢBᵢh_k² = ∑ᵢkᵢBᵢh_k²*k⁻²  # Φₖ
  
  # B  = B* - ∇Φ = Bᵢ - kᵢΦₖ  
  @. bxh  -= im*kᵢ.*∑ᵢkᵢBᵢh_k²
  @. byh  -= im*kⱼ.*∑ᵢkᵢBᵢh_k²
  @. bzh  -= im*kₖ.*∑ᵢkᵢBᵢh_k²
  
  #Update to Real Space vars
  ldiv!(vars.bx, grid.rfftplan, deepcopy(bxh))
  ldiv!(vars.by, grid.rfftplan, deepcopy(byh))
  ldiv!(vars.bz, grid.rfftplan, deepcopy(bzh))
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

  vars = prob.vars
  grid = prob.grid
  params = prob.params
  #find Φₖ
  kᵢ,kⱼ,kₖ = grid.kr,grid.l,grid.m
  k⁻² = grid.invKrsq
  @. vars.nonlin1  *= 0
  @. vars.nonlinh1 *= 0       
  ∑ᵢkᵢUᵢh_k² = vars.nonlinh1
  ∑ᵢkᵢUᵢ_k²  = vars.nonlin1
  @views uxh = prob.sol[:, :, :, params.ux_ind]
  @views uyh = prob.sol[:, :, :, params.uy_ind]
  @views uzh = prob.sol[:, :, :, params.uz_ind]
  @. ∑ᵢkᵢUᵢh_k² =  -im*(kᵢ*uxh + kⱼ*uyh + kₖ*uzh)
  @. ∑ᵢkᵢUᵢh_k² =  ∑ᵢkᵢUᵢh_k²*k⁻²  # Φₖ
  
  # B  = B* - ∇Φ = Bᵢ - kᵢΦₖ  
  @. uxh  -= im*kᵢ.*∑ᵢkᵢUᵢh_k²
  @. uyh  -= im*kⱼ.*∑ᵢkᵢUᵢh_k²
  @. uzh  -= im*kₖ.*∑ᵢkᵢUᵢh_k²
  
  #Update to Real Space vars
  ldiv!(vars.ux, grid.rfftplan, deepcopy(uxh))
  ldiv!(vars.uy, grid.rfftplan, deepcopy(uyh))
  ldiv!(vars.uz, grid.rfftplan, deepcopy(uzh))
end

end