# ----------
# Problem Generation Module : Taylor-Green vortex Dynamo from Nore et al, Physics of Plasmas, 4, 1 (1997);
# ----------

mutable struct N97_vars{Aphys,Atrans}
  fx  :: Aphys
  fy  :: Aphys
  fxh :: Atrans
  fyh :: Atrans
end

function N97ForceDriving!(N, sol, t, clock, vars, params, grid)
  #N97 Force 
  @views @. N[:,:,:,params.ux_ind] += vars.usr_vars.fxh;
  @views @. N[:,:,:,params.uy_ind] += vars.usr_vars.fyh;
end

function SetUpN97!(prob; F0 = 1, kf = 2)
  grid = prob.grid;
  x,y,z = grid.x,grid.y,grid.z;
  x = reshape(x,(size(x)[1],1,1));
  y = reshape(y,(1,size(x)[1],1));
  z = reshape(z,(1,1,size(x)[1]));
  nx,ny,nz = grid.nx,grid.ny,grid.nz;
  fx,fy = prob.vars.usr_vars.fx ,prob.vars.usr_vars.fy;
  fxh,fyh = prob.vars.usr_vars.fxh,prob.vars.usr_vars.fyh;
  fx_ = F0* sin.(kf*x).*cos.(kf*y).*cos.(kf*z);
  fy_ = F0*-cos.(kf*x).*sin.(kf*y).*cos.(kf*z);
  copyto!(fx,fx_);
  copyto!(fy,fy_);
  mul!(fxh,grid.rfftplan,fx);
  mul!(fyh,grid.rfftplan,fy);
  return nothing;
end

function GetN97vars_And_function(::Dev, nx::Int,ny::Int,nz::Int; T = Float32) where Dev
  @devzeros Dev         T  (            nx , ny, nz) fx  fy
  @devzeros Dev Complex{T} ( div(nx,2) + 1 , ny, nz) fxh fyh
    
  return  N97_vars(fx,fy,fxh,fyh), N97ForceDriving!;  
end