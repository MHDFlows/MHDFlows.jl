# ----------
# Problem Generation Module : A99 Turbulence Module from 
# ----------

mutable struct A99_vars{Atrans,T}
  A   :: T
  b   :: T
  Fk  :: Atrans
  e1x :: Atrans
  e1y :: Atrans
  e2x :: Atrans
  e2y :: Atrans
  e2z :: Atrans
  gi  :: Atrans
  eⁱᶿ :: Atrans
end

function GetA99vars_And_function(::Dev, nx::Int,ny::Int,nz::Int; T = Float32, C =false) where Dev

  A = convert(T,1.0);
  b = convert(T,1.0);
  @devzeros Dev Complex{T} ( div(nx,2) + 1 , ny, nz) Fk e1x e1y e2x e2y e2z gi eⁱᶿ
  A99 = A99_vars(A,b,Fk,e1x,e1y,e2x,e2y,e2z,gi,eⁱᶿ);

    
  if C
    return  A99, A99ForceDriving_Compressible!
  else
    return  A99, A99ForceDriving!;  
  end
end

function A99ForceDriving!(N, sol, t, clock, vars, params, grid)

  # A99 Force
  randN = typeof(N) <: Array ? Base.rand : CUDA.rand;
  T  = eltype(grid);
  A  = vars.usr_vars.A::T;
  b  = vars.usr_vars.b::T;
  Fk = vars.usr_vars.Fk; 
  e1x, e1y = vars.usr_vars.e1x,vars.usr_vars.e1y;
  e2x, e2y, e2z = vars.usr_vars.e2x,vars.usr_vars.e2y,vars.usr_vars.e2z;
  eⁱᶿ, gi =  vars.usr_vars.eⁱᶿ, vars.usr_vars.gi;
  Φ  = vars.nonlinh1;
    
  @. eⁱᶿ = exp(@.. im.*randN(T,grid.nkr,grid.nl,grid.nm)*2π);
  rand!(Φ); @. Φ*=π;
  @. gi  = -tanh(b*(Φ - π/2))/tanh(b*π/2);
  @. N[:,:,:,params.ux_ind] += A*Fk*eⁱᶿ*gi*e1x;
  @. N[:,:,:,params.uy_ind] += A*Fk*eⁱᶿ*gi*e1y;
    
  # Work out the second conponement
  @. eⁱᶿ .= exp(@.. im.*randN(T,grid.nkr,grid.nl,grid.nm)*2π);
  @. gi  = √(@.. 1 - gi^2); 
  @. N[:,:,:,params.ux_ind] += A*Fk*eⁱᶿ*gi*e2x;
  @. N[:,:,:,params.uy_ind] += A*Fk*eⁱᶿ*gi*e2y;
  @. N[:,:,:,params.uz_ind] += A*Fk*eⁱᶿ*gi*e2z;   

  return nothing
end

function A99ForceDriving_Compressible!(N, sol, t, clock, vars, params, grid)

  # A99 Force with the support of compressibiltiy
  randN = typeof(N) <: Array ? Base.rand : CUDA.rand;
  T  = eltype(grid);
  A  = vars.usr_vars.A::T;
  b  = vars.usr_vars.b::T;
  Fk = vars.usr_vars.Fk; 
  e1x, e1y = vars.usr_vars.e1x,vars.usr_vars.e1y;
  e2x, e2y, e2z = vars.usr_vars.e2x,vars.usr_vars.e2y,vars.usr_vars.e2z;
  eⁱᶿ, gi =  vars.usr_vars.eⁱᶿ, vars.usr_vars.gi;
  Φ  = vars.nonlinh1;
    
  # Work out the first conponement
  @. eⁱᶿ = exp(@.. im.*randN(T,grid.nkr,grid.nl,grid.nm)*2π);
  rand!(Φ); @. Φ*=π;
  @. gi  = -tanh(b*(Φ - π/2))/tanh(b*π/2);
  
  aᵢtoFᵢ!(view(N,:,:,:,params.ux_ind),A.*Fk.*eⁱᶿ.*gi.*e1x,vars,grid);
  aᵢtoFᵢ!(view(N,:,:,:,params.uy_ind),A.*Fk.*eⁱᶿ.*gi.*e1y,vars,grid);
    
  # Work out the second conponement
  @. eⁱᶿ = exp(@.. im.*randN(T,grid.nkr,grid.nl,grid.nm)*2π);
  @. gi  = √(1 - gi.^2); 
  aᵢtoFᵢ!(view(N,:,:,:,params.ux_ind),A.*Fk.*eⁱᶿ.*gi.*e2x,vars,grid);
  aᵢtoFᵢ!(view(N,:,:,:,params.uy_ind),A.*Fk.*eⁱᶿ.*gi.*e2y,vars,grid);
  aᵢtoFᵢ!(view(N,:,:,:,params.uz_ind),A.*Fk.*eⁱᶿ.*gi.*e2z,vars,grid);

  return nothing
end

function SetUpFk(prob; kf = 2, P = 1,σ²= 1)
  AT   = Array;
  grid = prob.grid;
  kx,ky,kz  = AT(grid.kr),AT(grid.l),AT(grid.m);
  Lx,Ly,Lz  = grid.Lx,grid.Ly,grid.Lz;
  dx,dy,dz  = grid.dx,grid.dy,grid.dz;
  k⁻¹  =  sqrt.(AT(grid.invKrsq));
  k    =  sqrt.(AT(grid.Krsq));
  k⊥   = @. √(kx^2 + ky^2);
  dk⁻² = @. 1/(k+1)^2;
  ∫Fkdk  = sum(@. exp(-(k.-kf)^2/σ²)*dk⁻²)
  A   = sqrt(P*3*(Lx/dx)*(Ly/dy)*(Lz/dz)/∫Fkdk*(1/dx/dy/dz));
  Fk  = @. A*√(exp(-(k.-kf)^2/σ²)/2/π)*k⁻¹;
  # Reason : https://github.com/FourierFlows/FourierFlows.jl/issues/326
  @. Fk[1,:,:] .= 0;
    
  e1x = @.  ky/k⊥;
  e1y = @. -kx/k⊥;
  e2x = @. kx*kz/k⊥*k⁻¹;
  e2y = @. ky*kz/k⊥*k⁻¹;
  e2z = @. -k⊥*k⁻¹;
    
  e1x[isnan.(e1x)] .= 0;
  e1y[isnan.(e1y)] .= 0;
  e2x[isnan.(e2x)] .= 0;
  e2y[isnan.(e2y)] .= 0;
  
  copyto!(prob.vars.usr_vars.Fk,  Fk);
  copyto!(prob.vars.usr_vars.e1x,e1x);
  copyto!(prob.vars.usr_vars.e1y,e1y);
  copyto!(prob.vars.usr_vars.e2x,e2x);
  copyto!(prob.vars.usr_vars.e2y,e2y);
  copyto!(prob.vars.usr_vars.e2z,e2z); 
    
end

function aᵢtoFᵢ!(∂pᵢ∂t,aᵢh,vars,grid)
  ρ  = vars.ρ;
  ldiv!(vars.nonlin1,grid.rfftplan,aᵢh);
  @. vars.nonlin1*=ρ;
  mul!(vars.nonlinh1,grid.rfftplan,vars.nonlin1);
  @. ∂pᵢ∂t += vars.nonlinh1;
  return nothing
end