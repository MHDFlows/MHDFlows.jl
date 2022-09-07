#-----------------------
# Negative damping turbulence module:
# We consider the following force Fᵢ for direction i :
# Fᵢ = fᵢ*uᵢ
#----------------------

mutable struct ND_vars{Float,Aphys}
  P   :: Float
  fx  :: Aphys
  fy  :: Aphys
  fz  :: Aphys
end

function SetUpND!(prob,P,fx,fy,fz)
  vars = prob.vars;
  vars.usr_vars.P = P;
  copyto!(vars.usr_vars.fx,fx);
  copyto!(vars.usr_vars.fy,fy);
  copyto!(vars.usr_vars.fz,fz);
  return nothing;
end

function NDForceDriving!(N, sol, t, clock, vars, params, grid)
  uvars = vars.usr_vars; 
  P   = vars.usr_vars.P;
  Fᵢ  = vars.nonlin1;
  Fᵢh = vars.nonlinh1;
  dx,dy,dz = grid.dx,grid.dy,grid.dz;
  ux,uy,uz = vars.ux,vars.uy,vars.uz;
  fx,fy,fz = uvars.fx,uvars.fy,uvars.fz;

  ∫uᵢfᵢdV = (sum(abs.(@. ux^2*fx)) + sum(abs.(@. uy^2*fy)) +
             sum(abs.(@. uz^2*fz)))*dx*dy*dz;
  A = P/∫uᵢfᵢdV;
  for (fᵢ,uᵢ,uᵢind) in zip([ux,uy,uz],
                           [fx,fy,fz],
                           [params.ux_ind,params.uy_ind,params.uz_ind])

    @. Fᵢh*=0;
    @. Fᵢ = fᵢ*uᵢ;
    mul!(Fᵢh, grid.rfftplan, Fᵢ);
    @views @. N[:,:,:,uᵢind] += A*Fᵢh;
  end
  return nothing;
end

function GetNDvars_And_function(::Dev, nx::Int,ny::Int,nz::Int; T = Float32) where Dev
  @devzeros Dev T  ( nx, ny, nz) fx  fy fz
    
  return  ND_vars(0.0 ,fx,fy,fz), NDForceDriving!;  
end