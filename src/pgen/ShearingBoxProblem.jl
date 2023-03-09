# ----------
# Problem Generation Module : Shearingbox Module
# ----------
function Setup_Shearingbox!(prob; q = 0.0, ν = 0.0)
  @assert prob.flag.s == true

  grid = prob.grid;
  T    = eltype(grid)
  params = prob.params;
  usr_params = params.usr_params;
  Lx,Ly = grid.Lx,grid.Ly;
  
  τΩ = abs(Lx/Ly/q)
  usr_params.ν  = T(ν)
  usr_params.τΩ = T(τΩ)
  usr_params.q  = T(q)
  copyto!(usr_params.ky₀ , grid.l)

  return nothing

end


function Get_shear_profile(grid,q::AbstractFloat,Ω::AbstractFloat)
  # U0 ≡ −qΩ \hat{y} - > - qΩ x 
  @devzeros typeof(CPU()) eltype(grid) (grid.nx, grid.ny, grid.nz) U₀x U₀y
  @. U₀x = - q*Ω;

  return U₀x,U₀y
end