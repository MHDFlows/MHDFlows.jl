# ----------
# General Function Module, providing function for setup IC of the problem
# ----------


"""
Construct a Cylindrical Mask Function χ for VP method
  Keyword arguments
=================
- `grid`: MHDFlows problem's grid
- `R₂` : Outwards radius boundary
- `R₁` : Inwards radius boundary
$(TYPEDFIELDS)
"""
function Cylindrical_Mask_Function(grid;R₂=0.82π,R₁=0.0π)
  nx,ny,nz = grid.nx,grid.ny,grid.nz
  x,y,z = grid.x,grid.y,grid.z
  S = BitArray(undef, nx::Int,ny::Int,nz::Int)
  
  for k ∈ 1:nz::Int, j ∈ 1:ny::Int,i ∈ 1:nx::Int
    xᵢ,yᵢ,zᵢ = x[i]::AbstractFloat,y[j]::AbstractFloat,z[k]::AbstractFloat
    Rᵢ       = √(xᵢ^2+yᵢ^2)
    # S = 0 if inside fluid domain while S = 1 in the solid domain 
    S[i,j,k] = (R₂ >= Rᵢ >= R₁) ?  0 : 1
  end    
  return S
end
"""
function of setting up the initial condition of the problem
  Keyword arguments
=================
- `prob`: MHDFlows problem
- `ux/uy/uz` : velocity in real space
- `bx/by/bz` : B-field in real space
- `U₀x/U₀y/U₀z/B₀x/B₀y/B₀z` : VP method parameter
$(TYPEDFIELDS)
"""
function SetUpProblemIC!(prob; ux = [], uy = [], uz =[],
                               bx = [], by = [], bz =[],
                               U₀x= [], U₀y= [], U₀z=[],
                               B₀x= [], B₀y= [], B₀z=[])
  sol = prob.sol
  vars = prob.vars
  grid = prob.grid
  params = prob.params;
  # Copy the data to both output and solution array
  for (uᵢ,prob_uᵢ,uᵢind) in zip([ux,uy,uz],[vars.ux,vars.uy,vars.uz],
                                [params.ux_ind,params.uy_ind,params.uz_ind])
    if uᵢ != []
      @views sol₀ =  sol[:, :, :, uᵢind]
      copyto!(prob_uᵢ,uᵢ)
      mul!(sol₀ , grid.rfftplan, prob_uᵢ)
    end
  end
  if prob.flag.b 
    for (bᵢ,prob_bᵢ,bᵢind) in zip([bx,by,bz],[vars.bx,vars.by,vars.bz],
                                  [params.bx_ind,params.by_ind,params.bz_ind])
      if bᵢ != []
        @views sol₀ =  sol[:, :, :, bᵢind]
        copyto!(prob_bᵢ,bᵢ)
        mul!(sol₀ , grid.rfftplan, prob_bᵢ)
      end
    end
  end
  if prob.flag.vp
    for (Uᵢ,prob_Uᵢ) in zip([U₀x,U₀y,U₀z],
                            [params.U₀x,params.U₀y,params.U₀z])
      Uᵢ == [] ? nothing : copyto!(prob_Uᵢ,Uᵢ)
    end

    if prob.flag.b
      for (Bᵢ,prob_Bᵢ) in zip([B₀x,B₀y,B₀z],
                              [params.B₀x,params.B₀y,params.B₀z])
        Bᵢ == [] ? nothing : copyto!(prob_Bᵢ,Bᵢ)
      end
    end
  end
  return nothing;
  
end