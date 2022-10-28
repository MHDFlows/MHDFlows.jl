# ----------
# Problem Generation Module : Cho(2001) Turbulence Module
# ----------

mutable struct Cho_vars{Atrans,T}
  Fk  :: Atrans
  s1y :: Atrans 
  s1z :: Atrans 
  s2x :: Atrans 
  s2y :: Atrans 
  s2z :: Atrans 
  Φ1  :: Atrans
  Φ2  :: Atrans
  kf  :: T
  P   :: T
end

function Get_Cho_vars_and_function(::Dev, nx::Int, ny::Int, nz::Int; T=Float32) where Dev
  @devzeros Dev Complex{T}  ( div(nx,2)+1, ny, nz) Φ1 Φ2
  @devzeros Dev Complex{T}  ( div(nx,2)+1, ny, nz) Fk s1y s1z s2x s2y s2z;
  return Cho_vars(Fk,s1y,s1z,s2x,s2y,s2z,Φ1,Φ2,T(0.0),T(0.0)), ChoForceDriving!;
end

#=function ChoForceDriving!(N, sol, t, clock, vars, params, grid)
  # Define the parameter from vars
  T  = eltype(grid);
  eⁱᶿ = vars.nonlinh1;
  Φ1,Φ2,Fk = vars.usr_vars.Φ1,vars.usr_vars.Φ2,vars.usr_vars.Fk;
  s1y, s1z = vars.usr_vars.s1y,vars.usr_vars.s1z;
  s2x, s2y, s2z = vars.usr_vars.s2x,vars.usr_vars.s2y,vars.usr_vars.s2z;
  A,kf = copy(vars.usr_vars.P::T),vars.usr_vars.kf::T;
  vi = params.ν;
  dt = clock.dt;
  A*= exp(-vi*kf^2*dt);

  # Actual computation
  @. eⁱᶿ = exp.(im*(Φ1.+Φ2)./2);
  @. N[:,:,:,params.ux_ind] += A.*eⁱᶿ.*( 0 .*cos.((Φ1.-Φ2)/2) + s2x.*sin.((Φ1.-Φ2)/2));
  @. N[:,:,:,params.uy_ind] += A.*eⁱᶿ.*(s1y.*cos.((Φ1.-Φ2)/2) + s2y.*sin.((Φ1.-Φ2)/2));
  @. N[:,:,:,params.uz_ind] += A.*eⁱᶿ.*(s1z.*cos.((Φ1.-Φ2)/2) + s2z.*sin.((Φ1.-Φ2)/2));
  # Large Scale Forcing
  @. N[:,:,:,params.ux_ind] += Fk;
  return nothing;
end=#

function ChoForceDriving!(N, sol, t, clock, vars, params, grid)
  # Define the parameter from vars
  T  = eltype(grid);
  eⁱᶿ = vars.nonlinh1;
  Φ1,Φ2,Fk = vars.usr_vars.Φ1,vars.usr_vars.Φ2,vars.usr_vars.Fk;
  s1y, s1z = vars.usr_vars.s1y,vars.usr_vars.s1z;
  s2x, s2y, s2z = vars.usr_vars.s2x,vars.usr_vars.s2y,vars.usr_vars.s2z;
  A,kf = copy(vars.usr_vars.P::T),vars.usr_vars.kf::T;
  vi = params.ν;
  dt = clock.dt;
  A*= exp(-vi*kf^2*dt);

  # Actual computation
  @. eⁱᶿ = 0;
  @. eⁱᶿ = exp.(im*(Φ1.+Φ2)./2);
  for (u_ind,s1,s2) in zip([params.ux_ind,params.uy_ind,params.uz_ind],[0,s1y,s1z],[s2x,s2y,s2z])
       @. N[:,:,:,u_ind] += A.*eⁱᶿ.*( s1 .*cos.((Φ1.-Φ2)/2) + s2.*sin.((Φ1.-Φ2)/2));   
  end

  # Large Scale Forcing
  if minimum(mean(vars.ux,dims=1)[:]) > -1.0
    @. N[:,:,:,params.ux_ind] += Fk;
  end
  return nothing;
end


function Set_up_Cho_vars(prob; P = 1e7, kf = 15)
  # Define the parameter will be used
  grid =  prob.grid;
  vars = prob.vars;
  nx,ny,nz  = grid.nx,grid.ny,grid.nz;
  Lx,Ly,Lz  = grid.Lx,grid.Ly,grid.Lz;
  dx,dy,dz  = grid.dx,grid.dy,grid.dz;
  kx,ky,kz  = grid.kr,grid.l,grid.m;
  T = eltype(grid);
  # The 22 conponment 
  k_component = 22;
  fox,foy,foz = zeros(Int32,k_component),zeros(Int32,k_component),zeros(Int32,k_component);
  k = 1;
  for θ ∈ [15,20,25].*π/180 #anisotropic turbulence
    for ϕ ∈ [-25,-15,-5,0,5,15,25].*π/180
      fox[k] = round(Int32,kf*cos(θ));
      foy[k] = round(Int32,kf*sin(θ)*sin(ϕ));
      foz[k] = round(Int32,kf*sin(θ)*cos(ϕ));
      k+=1;
    end
  end
  fox[22],foy[22],foz[22] = kf,0.0,0.0;

  # Set up vector set s1 s2 that ⊥ k_f
  @devzeros typeof(CPU()) Complex{T} ( div(nx,2)+1, ny, nz) Φ1 Φ2
  @devzeros typeof(CPU()) Complex{T} ( div(nx,2)+1, ny, nz) Fk s1y s1z s2x s2y s2z

  kr,l,m = Array(grid.kr)[:],Array(grid.l)[:],Array(grid.m)[:];
  dx,dy,dz = grid.dx,grid.dy,grid.dz;
  for k_i = 1:k_component
    # index 1,2,3 -> i,j,k direction
    rkx,rky,rkz = fox[k_i],foy[k_i],foz[k_i];
    kx = findall(kr .== rkx)[1];
    ky = findall(l  .== rky)[1];
    kz = findall(m  .== rkz)[1];
    ryz = √( rky^2 + rkz^2 );
    rxyz= √( rkx^2 + rky^2 +rkz^2);
    if (ryz == 0.0) || (rxyz == 0.0)
      s1y[kx,ky,kz] = 0.0
      s1z[kx,ky,kz] = 1.0
      s2x[kx,ky,kz] = 0.0
      s2y[kx,ky,kz] = 0.0
      s2z[kx,ky,kz] = 1.0
    else
      s1y[kx,ky,kz] =  rkz / ryz
      s1z[kx,ky,kz] = -rky / ryz
      s2x[kx,ky,kz] = -ryz / rxyz
      s2y[kx,ky,kz] =  rkx*rky / rxyz / ryz
      s2z[kx,ky,kz] =  rkx*rkz / rxyz / ryz
    end
  end

  Fk[1,2,1] = -10/dx/dy/dz;
  #Fk[1,1,1] =  0/dx/dy/dz;
  copyto!(prob.vars.usr_vars.Fk , Fk);
  copyto!(prob.vars.usr_vars.s1y,s1y);
  copyto!(prob.vars.usr_vars.s1z,s1z);
  copyto!(prob.vars.usr_vars.s2x,s2x);
  copyto!(prob.vars.usr_vars.s2y,s2y);
  copyto!(prob.vars.usr_vars.s2z,s2z);

  # Work out the Φ conponement
  randN = typeof(vars.usr_vars.Φ1) <: Array ? Base.rand : CUDA.rand;
  Φ1 =  rand(T,grid.nkr,grid.nl,grid.nm).*2π .+ 0*im;
  Φ2 =  rand(T,grid.nkr,grid.nl,grid.nm).*2π .+ 0*im;
  copyto!(vars.usr_vars.Φ1,Φ1);
  copyto!(vars.usr_vars.Φ2,Φ2);

  # Work the Amp of A
  k     = @. √(grid.Krsq);
  k⊥   = @. √(kx^2 + ky^2);
  dk⁻²  = @. 1/(k+1)^2;
  F = 0 .*vars.nonlinh1;
  F[abs.(s1y).>0] .= 1;
  F[abs.(s1z).>0] .= 1;
  ∫Fkdk  = sum(@. F*dk⁻²)
  A   = sqrt(P*3*(Lx/dx)*(Ly/dy)*(Lz/dz)/∫Fkdk*(1/dx/dy/dz));
  vars.usr_vars.P = A;

  return nothing;
end

function Random_iterator!(prob)
  #random generator ∈ [-1,1]
  vars = prob.vars;
  grid = prob.grid;
  Rand = typeof(vars.usr_vars.Φ1) <: Array ? Base.rand : CUDA.rand;
  randN(T,nx,ny,nz) = 2 .*(Rand(T,nx,ny,nz) .- 0.5);
  
  T = eltype(prob.grid);
  Φ1,Φ2 = vars.usr_vars.Φ1,vars.usr_vars.Φ2;
  Φ_changefraction   = convert(T,0.02);

  # For each time step, slowly changing the amplitude or phase by 1 or 2%
  copyto!(Φ1, Φ1.*( 1 .+ 2*π .*randN(T,grid.nkr,grid.nl,grid.nm).*Φ_changefraction));
  copyto!(Φ2, Φ2.*( 1 .+ 2*π .*randN(T,grid.nkr,grid.nl,grid.nm).*Φ_changefraction));
  return nothing
end  


function SetUpFk_(prob; kf = [2], P = 1,σ²= 1,Rᵢ = [1.0])
  grid = prob.grid;
  kx,ky,kz = grid.kr,grid.l,grid.m;
  Lx,Ly,Lz  = grid.Lx,grid.Ly,grid.Lz;
  dx,dy,dz  = grid.dx,grid.dy,grid.dz;
  k⁻¹  = @. √(grid.invKrsq);
  k    = @. √(grid.Krsq);
  k⊥   = @. √(kx^2 + ky^2);
  dk⁻² = @. 1/(k+1)^2;
  ∫Fkdk = 0;
  for kfᵢ in kf
    ∫Fkdk  += sum(@. exp(-(k.-kfᵢ)^2/σ²)*dk⁻²)
  end

  A   = sqrt(P*3*(Lx/dx)*(Ly/dy)*(Lz/dz)/∫Fkdk*(1/dx/dy/dz));
  Fk = 0 .*copy(k);
  for (kfᵢ,R) in zip(kf,Rᵢ)
    @. Fk  += R*A*√(exp(-(k.-kfᵢ)^2/σ²)/2/π)*k⁻¹;
  end
  
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