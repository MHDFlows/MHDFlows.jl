# ----------
# Geometry Module, providing Geometry convertion function
# ----------


# xyz Coordinetes -> rθz Coordinates
function xy_to_polar(ux,uy;Lx=2π,Ly=Lx,T=Float32)   
  nx,ny,nz = size(ux);  
  dev = CPU();
  grid = TwoDGrid(dev, nx, Lx, ny, Ly; T=T)
  Ur,Uθ = xy_to_polar(ux,uy,grid;Lx=2π,Ly=Lx,T=Float32);
  return Ur,Uθ;
end

function xy_to_polar(ux::Array,uy::Array,grid;Lx=2π,Ly=Lx,T=Float32)
#=
  Function for converting x-y vector to r-θ vector, using linear transform
    [x']  =  [cos(θ) -rsin(θ)][r']
    [y']     [sin(θ)  rcos(θ)][θ']
    So e_r =  cosθ ̂i + sinθ ̂j
       e_θ = -sinθ ̂j + cosθ ̂j
=#    
  nx,ny,nz = size(ux);  
  Ur = zeros(T,nx,ny,nz);
  Uθ = zeros(T,nx,ny,nz);
  for j ∈ 1:ny, i ∈ 1:nx
    r = sqrt(grid.x[i]^2+grid.y[j]^2);
    θ = atan(grid.y[j],grid.x[i]) ;
    θ = isnan(θ) ? π/2 : θ;
    sinθ = sin(θ);
    cosθ = cos(θ);    
    Uθ[i,j,:] .= @. -sinθ*ux[i,j,:] + cosθ*uy[i,j,:];    
    Ur[i,j,:] .= @.  cosθ*ux[i,j,:] + sinθ*uy[i,j,:];    
  end
  return Ur,Uθ;
end
