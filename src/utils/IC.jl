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
  nx,ny,nz = grid.nx,grid.ny,grid.nz;
  x,y,z = grid.x,grid.y,grid.z;
  S = BitArray(undef, nx::Int,ny::Int,nz::Int);
  
  for k ∈ 1:nz::Int, j ∈ 1:ny::Int,i ∈ 1:nx::Int
    xᵢ,yᵢ,zᵢ = x[i]::AbstractFloat,y[j]::AbstractFloat,z[k]::AbstractFloat;
    Rᵢ       = √(xᵢ^2+yᵢ^2);
    # S = 0 if inside fluid domain while S = 1 in the solid domain 
    S[i,j,k] = (R₂ >= Rᵢ >= R₁) ?  0 : 1; 
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
  sol = prob.sol;
  vars = prob.vars;
  grid = prob.grid;
  params = prob.params;
  # Copy the data to both output and solution array
  for (uᵢ,prob_uᵢ,uᵢind) in zip([ux,uy,uz],[vars.ux,vars.uy,vars.uz],
                                [params.ux_ind,params.uy_ind,params.uz_ind])
    if uᵢ != []
      @views sol₀ =  sol[:, :, :, uᵢind];
      copyto!(prob_uᵢ,uᵢ);
      mul!(sol₀ , grid.rfftplan, prob_uᵢ);
    end
  end
  if prob.flag.b 
    for (bᵢ,prob_bᵢ,bᵢind) in zip([bx,by,bz],[vars.bx,vars.by,vars.bz],
                                  [params.bx_ind,params.by_ind,params.bz_ind])
      if bᵢ != []
        @views sol₀ =  sol[:, :, :, bᵢind];
        copyto!(prob_bᵢ,bᵢ);
        mul!(sol₀ , grid.rfftplan, prob_bᵢ);
      end
    end
  end
  if prob.flag.vp
    for (Uᵢ,prob_Uᵢ) in zip([U₀x,U₀y,U₀z],
                            [params.U₀x,params.U₀y,params.U₀z])
      Uᵢ == [] ? nothing : copyto!(prob_Uᵢ,Uᵢ);
    end

    if prob.flag.b
      for (Bᵢ,prob_Bᵢ) in zip([B₀x,B₀y,B₀z],
                              [params.B₀x,params.B₀y,params.B₀z])
        Bᵢ == [] ? nothing : copyto!(prob_Bᵢ,Bᵢ);
      end
    end
  end
  return nothing;
  
end


"""
Construct a Div Free Spectra Vector Map with power-law relation
  Keyword arguments
=================
- `Nx/Ny/Nz`: size of the Vector Map
- `k0` : Slope of the Map 
- `b`  : Anisotropy of the Map
$(TYPEDFIELDS)
"""
function DivFreeSpectraMap( Nx::Int, Ny::Int, Nz::Int;
                            Lx = 2π,
                            dev = CPU(), 
                            P = 1, k0 = -5/3/2, b = 1, T = Float64)
  grid = ThreeDGrid(dev; nx = Nx, Lx = Lx, ny=Ny, nz=Nz, T = T);
  return DivFreeSpectraMap( grid; P = P, k0 = k0, b = b);
end

function DivFreeSpectraMap( grid;
                            P = 1, k0 = -5/3/2, b = 1)
    
  T = eltype(grid);  
  @devzeros typeof(grid.device) Complex{T} (grid.nkr,grid.nl,grid.nm) eⁱᶿ Fk Fxh Fyh Fzh 
  @devzeros typeof(grid.device)         T  (grid.nx ,grid.ny,grid.nz) Fx Fy Fz
  
  kx,ky,kz  = grid.kr,grid.l,grid.m;  
  Lx,Ly,Lz  = grid.Lx,grid.Ly,grid.Lz;
  dx,dy,dz  = grid.dx,grid.dy,grid.dz;
  k⁻¹  = @. √(grid.invKrsq);
  k    = @. √(grid.Krsq);
  k⊥   = @. √(kx^2 + ky^2);
  dk⁻² = @. 1/(k+1)^2;
  Fk   = @. k.^(k0);
  CUDA.@allowscalar Fk[1,1,1] = 0.0;

  ∫Fkdk  = sum(@. Fk*dk⁻²);
  A   = sqrt(P*3*(Lx/dx)*(Ly/dy)*(Lz/dz)/∫Fkdk*(1/dx/dy/dz));
  Fk*=A;
    
  e1x = @.  ky/k⊥;
  e1y = @. -kx/k⊥;
  e2x = @. kx*kz/k⊥*k⁻¹;
  e2y = @. ky*kz/k⊥*k⁻¹;
  e2z = @. -k⊥*k⁻¹;
  e1x[isnan.(e1x)] .= 0;
  e1y[isnan.(e1y)] .= 0;
  e2x[isnan.(e2x)] .= 0;
  e2y[isnan.(e2y)] .= 0;
    
  # Work out the random conponement 
  randN = grid.device == CPU() ? rand : CUDA.rand;
  eⁱᶿ .= exp.(im.*randN(T,grid.nkr,grid.nl,grid.nm)*2π);
  @. Fxh += Fk*eⁱᶿ*e2x;
  @. Fyh += Fk*eⁱᶿ*e2y;
  @. Fzh += Fk*eⁱᶿ*e2z;
  
  dealias!(Fxh, grid)
  dealias!(Fyh, grid)
  dealias!(Fzh, grid)

  ldiv!(Fx, grid.rfftplan, deepcopy(Fxh));  
  ldiv!(Fy, grid.rfftplan, deepcopy(Fyh));
  ldiv!(Fz, grid.rfftplan, deepcopy(Fzh));

  return Fx,Fy,Fz;
  
end
