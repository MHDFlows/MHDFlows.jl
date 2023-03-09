# ----------
# Implicit timeStepper for EMHD simulation (Harned & Mikic, 1989, J. Computational Phys,  83, 1, pp. 1-15)
# ----------

struct HM89TimeStepper{T,TL} <: FourierFlows.AbstractTimeStepper{T}
  F₀  :: T
  F₁  :: T
  B⁰  :: T
  B¹  :: T
  Bⁿ  :: T
   c  :: TL
end

function HM89TimeStepper(equation, dev::Device=CPU())
  @devzeros typeof(dev) equation.T equation.dims  F₀  F₁  B⁰  B¹ Bⁿ 

  c = (1//3, 15//16, 8//15)

  return HM89TimeStepper( F₀, F₁, B⁰, B¹, Bⁿ , c)
end

function stepforward!(sol, clock, ts::HM89TimeStepper, equation, vars, params, grid)
  HM89substeps!( sol, clock, ts, equation, vars, params, grid)

  clock.t += clock.dt
  
  clock.step += 1
  
  return nothing
end

function HM89substeps!(sol, clock, ts, equation, vars, params, grid)
  # we solve the equation of 
  # B^{n+1} + (Δt)²(B₀·∇)²∇^2B^{n+1} = Bⁿ + (Δt)²(B₀·∇)^2∇^2Bⁿ - Δt∇×(J^{n+1/2}× B^{n+1/2})
  # using fix point method
  square_mean(A,B,C) =  mapreduce((x,y,z)->√(x*x+y*y+z*z),max,A,B,C)

  t, Δt, c  = clock.t, clock.dt, ts.c
  
  B⁰, B¹, Bⁿ = ts.B⁰, ts.B¹, ts.Bⁿ

  ΔBh, B₀∇⁴B, ∇XJXB =  ts.F₀, ts.F₀, ts.F₁

  ΔBx, ΔBy, ΔBz = vars.bx, vars.by, vars.bz

  # check the mean field condition &  determine the k₀ for later usage
  mB = ( mean(vars.bx), mean(vars.by), mean(vars.bz) )
  i = findmax(mB)[2]
  checkmB( mB, i )
  B₀ = mB[i] 
  k₀ = ifelse( i == 1, grid.kr, ifelse(i == 2,  grid.l, grid.m ) )
  
  # get B\^{n+1} guess from RK3 Method
  copyto!(B⁰, sol)
  LSRK3substeps!(sol, clock, ts, equation, vars, params, grid)
  copyto!( B¹,  sol)
  dealias!(B¹, grid)
  B_half = sol

  ε  = 1.0;
  err = 1e-4;

  while ε > err 
    
    @. B_half = (B⁰ + B¹)*0.5
    
    # get the ∇×(J × B) term
    equation.calcN!(∇XJXB, B_half, t, clock, vars, params, grid)
    
    # hyper diffusion term
    @. B_half = (B⁰ - B¹)
    hyperdiffusionterm!(B₀∇⁴B, B_half, B₀, k₀, grid)

    # get the term B\^ n + 1
    @. Bⁿ = B⁰ + Δt^2*B₀∇⁴B - Δt*∇XJXB
    dealias!(Bⁿ, grid)

    # compute the error
    @. ΔBh = (Bⁿ - B¹)
    ldiv!( ΔBx, grid.rfftplan, deepcopy( @view ΔBh[:,:,:,1] ) )
    ldiv!( ΔBy, grid.rfftplan, deepcopy( @view ΔBh[:,:,:,2] ) )
    ldiv!( ΔBz, grid.rfftplan, deepcopy( @view ΔBh[:,:,:,3] ) ) 
    ε = square_mean(ΔBx, ΔBy, ΔBz)

    # copy to Bⁿ to be B¹
    copyto!(B¹, Bⁿ)

  end

  copyto!(sol, B¹)
  RK3diffusion!(sol, ts, clock, vars, params, grid)
  DivFreeCorrection!(sol, vars, params, grid)

  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]))
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]))
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])) 

  return nothing

end

function DivFreeCorrection!(sol, vars, params, grid)
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

  #find Φₖ
  kᵢ,kⱼ,kₖ = grid.kr,grid.l,grid.m;
  k⁻² = grid.invKrsq;
  @. vars.nonlin1  *= 0;
  @. vars.nonlinh1 *= 0;       
  ∑ᵢkᵢBᵢh_k² = vars.nonlinh1;
  ∑ᵢkᵢBᵢ_k²  = vars.nonlin1;

  # it is N not sol
  @views bxh = sol[:, :, :, params.bx_ind];
  @views byh = sol[:, :, :, params.by_ind];
  @views bzh = sol[:, :, :, params.bz_ind];

  @. ∑ᵢkᵢBᵢh_k² = -im*(kᵢ*bxh + kⱼ*byh + kₖ*bzh);
  @. ∑ᵢkᵢBᵢh_k² = ∑ᵢkᵢBᵢh_k²*k⁻²;  # Φₖ
 
  # B  = B* - ∇Φ = Bᵢ - kᵢΦₖ  
  @. bxh  -= im*kᵢ.*∑ᵢkᵢBᵢh_k²;
  @. byh  -= im*kⱼ.*∑ᵢkᵢBᵢh_k²;
  @. bzh  -= im*kₖ.*∑ᵢkᵢBᵢh_k²;

  return nothing
end

function hyperdiffusionterm!(B₀∇⁴B, B, B₀, k₀, grid)
  #
  # hyper diffusion term from HM89
  #

  k² = grid.Krsq
  @. B₀∇⁴B = B₀*k₀^2*k²*B*2.5e-1
  
  return nothing
end

function checkmB(mB, i)
  @assert length(mB) == 3

  for j = 1:3
    if  i != j && mB[j] > 0.3
      error(" Only support single mean field driection! \n")
    end
  end

  return nothing
end

function LSRK3substeps!(sol, clock, ts, equation, vars, params, grid)
  # Low stoage 3 step RK3 method (LSRK3)
  # F0 = dt F(0)
  # p1 = p0 + c1 F0
  # F1  = dt*F(1) - F0*5/9
  # p2 = p0 + c2 F1
  # F2 = -153/128*F(1) + dt*F(2)
  # p3 = p2 + c3*F2

  t  = clock.t
  dt = clock.dt
  c  = ts.c

  equation.calcN!(ts.F₀, sol, t + dt, clock, vars, params, grid)
  @. ts.F₀ *=  dt
  @.  sol  += ts.F₀*c[1]*dt

  equation.calcN!(ts.F₁, sol, t + dt, clock, vars, params, grid)
  @. ts.F₁ *=  dt
  @. ts.F₁ -=  5/9*ts.F₀
  @.  sol  +=  c[2]*ts.F₁

  # reuse F2 = F0
  equation.calcN!(ts.F₀, sol, t + dt, clock, vars, params, grid)
  @. ts.F₀ *= dt
  @. ts.F₀ -= 153/128*ts.F₁
  @.   sol += c[3]*ts.F₀
  return nothing
end

function RK3diffusion!(sol, ts, clock, vars, params, grid)
  # LSKR3 for diffusion term 
  # F0 = dt F(0)
  # p1 = p0 + c1 F0
  # F1  = dt*F(1) - F0*5/9
  # p2 = p0 + c2 F1
  # F2 = -153/128*F(1) + dt*F(2)
  # p3 = p2 + c3*F2

  t  = clock.t
  dt = clock.dt
  c  = ts.c
  k² = grid.Krsq
  η  = params.η

  @. ts.F₀ =  -η*k²*sol
  @. ts.F₀ *=  dt
  @.  sol  += ts.F₀*c[1]*dt

  @. ts.F₁ =  -η*k²*sol
  @. ts.F₁ *=  dt
  @. ts.F₁ -=  5/9*ts.F₀
  @.  sol  +=  c[2]*ts.F₁

  # reuse F2 = F0
  @. ts.F₀ =  -η*k²*sol
  @. ts.F₀ *= dt
  @. ts.F₀ -= 153/128*ts.F₁
  @.   sol += c[3]*ts.F₀

  return nothing
end


