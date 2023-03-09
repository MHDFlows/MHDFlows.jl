# ----------
# Problem Generation Module : A99 Turbulence Module (Low Storage Version)
# ----------

mutable struct A99LS_vars{Atrans,T}
  A   :: T
  b   :: T
  Fke2x :: Atrans
  Fke2y :: Atrans
  Fke2z :: Atrans
end

function GetA99LSvars_And_function(::Dev, nx::Int,ny::Int,nz::Int; T = Float32, C =false) where Dev

  A = convert(T,1.0);
  b = convert(T,1.0);
  @devzeros Dev Complex{T} ( div(nx,2) + 1 , ny, nz) e2x e2y e2z
  A99 = A99LS_vars(A,b,e2x,e2y,e2z);

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
  Fke2x, Fke2y, Fke2z = vars.usr_vars.Fke2x,vars.usr_vars.Fke2y,vars.usr_vars.Fke2z;
  eⁱᶿ =  vars.nonlinh1;

  @. eⁱᶿ .= exp(@.. im.*randN(T,grid.nkr,grid.nl,grid.nm)*2π);
  @. N[:,:,:,params.ux_ind] += A*Fke2x*eⁱᶿ;
  @. N[:,:,:,params.uy_ind] += A*Fke2y*eⁱᶿ;
  @. N[:,:,:,params.uz_ind] += A*Fke2z*eⁱᶿ;

  return nothing
end

function A99ForceDriving_Compressible!(N, sol, t, clock, vars, params, grid)

  # A99 Force with the support of compressibiltiy
  randN = typeof(N) <: Array ? Base.rand : CUDA.rand;
  T  = eltype(grid);
  A  = vars.usr_vars.A::T;
  b  = vars.usr_vars.b::T;
  Fke2x, Fke2y, Fke2z = vars.usr_vars.Fke2x,vars.usr_vars.Fke2y,vars.usr_vars.Fke2z;
  eⁱᶿ =  vars.nonlinh2;
    
  @. eⁱᶿ .= exp(@.. im.*randN(T,grid.nkr,grid.nl,grid.nm)*2π);
  for (u_ind,Fki) in zip((params.ux_ind,params.uy_ind,params.uz_ind),(Fke2x,Fke2y,Fke2z))
    aᵢtoFᵢ!(view(N,:,:,:,u_ind), @.(A*Fki*eⁱᶿ) , vars, grid);
  end
  return nothing
end

function SetUpLSFk(prob; kf = 2, P = 1,σ²= 1)
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
  
  e2x = @. kx*kz/k⊥*k⁻¹;
  e2y = @. ky*kz/k⊥*k⁻¹;
  e2z = @. -k⊥*k⁻¹;
    
  e2x[isnan.(e2x)] .= 0;
  e2y[isnan.(e2y)] .= 0;
  e2z[isnan.(e2z)] .= 0;
  
  copyto!(prob.vars.usr_vars.Fke2x, @.. Fk*e2x);
  copyto!(prob.vars.usr_vars.Fke2y, @.. Fk*e2y);
  copyto!(prob.vars.usr_vars.Fke2z, @.. Fk*e2z); 
  return nothing;
end

function aᵢtoFᵢ!(∂pᵢ∂t,aᵢh,vars,grid)
  ρ  = vars.ρ;
  ldiv!(vars.nonlin1, grid.rfftplan, aᵢh);
  @. vars.nonlin1*=ρ;
  mul!(vars.nonlinh1, grid.rfftplan, vars.nonlin1);
  @. ∂pᵢ∂t += vars.nonlinh1;
  return nothing
end