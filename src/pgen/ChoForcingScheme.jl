# ----------
# Problem Generation Module : Cho(2001) Turbulence Module
# ----------

mutable struct Cho_vars{Aphys,Atrans,T}
  s1y :: Aphys 
  s1z :: Aphys 
  s2x :: Aphys 
  s2y :: Aphys 
  s2z :: Aphys 
  Φ1  :: Atrans
  Φ2  :: Atrans
  kf  :: T
  P   :: T
end

function Get_Cho_vars_and_function(::Dev, nx::Int, ny::Int, nz::Int; T=Float32) where Dev
  @devzeros Dev Complex{T} ( div(nx,2)+1, ny, nz) Φ1 Φ2
  @devzeros Dev         T  ( div(nx,2)+1, ny, nz) s1y s1z s2x s2y s2z
  return Cho_vars(s1,s2,Φ1,Φ2,0.0,0.0), ChoForcedriving;
end

function ChoForceDriving!(N, sol, t, clock, vars, params, grid)
  # Define the parameter from vars
  T  = eltype(grid);
  eⁱᶿ = vars.nonlinh1;
  Φ1,Φ2,Fk = vars.usr_vars.Φ1,vars.usr_vars.Φ2,vars.usr_vars.Fk;
  s1y, s1z = vars.usr_vars.s1y,vars.usr_vars.s1z;
  s2x, s2y, s2z = vars.usr_vars.s2x,vars.usr_vars.s2y,vars.usr_vars.s2z;
  A,kf = vars.usr_vars.b::T,vars.usr_vars.kf::T;
  vi = params.vi;
  dt= clock.dt;
  A*= exp(-vi*kf^2*dt);

  # Actual computation
  @. eⁱᶿ = exp.(im*(Φ₁.+Φ₂)./2);
  @. N[:,:,:,ux_ind] += A.*eⁱᶿ.*Fk.*(  0.*cos.((Φ1.-Φ2)/2) + s2x.*sin.((Φ1 .-Φ2)/2));
  @. N[:,:,:,uy_ind] += A.*eⁱᶿ.*Fk.*(s1y.*cos.((Φ1.-Φ2)/2) + s2y.*sin.((Φ1 .-Φ2)/2));
  @. N[:,:,:,uz_ind] += A.*eⁱᶿ.*Fk.*(s1z.*cos.((Φ1.-Φ2)/2) + s2z.*sin.((Φ1 .-Φ2)/2));

  return nothing;
end

function Set_up_Cho_vars(prob; kf = 15)
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
  @devzeros CPU() Complex{T} ( div(nx,2)+1, ny, nz) Φ1 Φ2
  @devzeros CPU()         T  ( div(nx,2)+1, ny, nz) s1y s1z s2x s2y s2z

  for k_i = 1:k_component
    # index 1,2,3 -> i,j,k direction
    rkx,rky,rkz = fox[k_i],foy[k_i],foz[k_i];
    ryz = √( rky^2 +rkz^2 );
    rxyz= √( rkx^2 +rky^2 +rkz^2);
    if (ryz == 0.0)
      s1y[fox[k_i],foy[k_i],foz[k_i]] = 0.0
      s1z[fox[k_i],foy[k_i],foz[k_i]] = 1.0
      s2x[fox[k_i],foy[k_i],foz[k_i]] = 0.0
      s2y[fox[k_i],foy[k_i],foz[k_i]] = 0.0
      s2z[fox[k_i],foy[k_i],foz[k_i]] = 1.0
    else
      s1y[fox[k_i],foy[k_i],foz[k_i]] =  rkz / ryz
      s1z[fox[k_i],foy[k_i],foz[k_i]] = -rky / ryz
      s2x[fox[k_i],foy[k_i],foz[k_i]] = -ryz / rxyz
      s2y[fox[k_i],foy[k_i],foz[k_i]] =  rkx*rky / rxyz / ryz
      s2z[fox[k_i],foy[k_i],foz[k_i]] =  rkx*rkz / rxyz / ryz
    end
  end

  copyto!(prob.vars.usr_vars.s1y,s1y);
  copyto!(prob.vars.usr_vars.s1z,s1z);
  copyto!(prob.vars.usr_vars.s2x,s2x);
  copyto!(prob.vars.usr_vars.s2y,s2y);
  copyto!(prob.vars.usr_vars.s2z,s2z);

  # Work out the first conponement
  randN = typeof(vars.usr_vars.Φ1) <: Array ? Base.rand : CUDA.rand;
  grid = prob.grid;
  @. vars.usr_vars.Φ1 = exp.(im.*randN(T,grid.nkr,grid.nl,grid.nm)*2π);
  @. vars.usr_vars.Φ2 = exp.(im.*randN(T,grid.nkr,grid.nl,grid.nm)*2π);  

  return nothing;
end