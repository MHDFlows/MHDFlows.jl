# ----------
# General Function Module, providing function for setup IC of the problem
# ----------

"""
    Cylindrical_Mask_Function(grid)

Construct a Cylindrical Mask Function χ for VP method
  Keyword arguments
=================
- `grid`: MHDFlows problem's grid
- `R₂` : Outwards radius boundary
- `R₁` : Inwards radius boundary
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
    SetUpProblemIC!(prob)

function of setting up the initial condition of the problem
  Keyword arguments
=================
- `prob`: MHDFlows problem
- `ρ`        : density in real space 
- `ux/uy/uz` : velocity in real space
- `bx/by/bz` : B-field in real space
- `U₀x/U₀y/U₀z/B₀x/B₀y/B₀z` : VP method parameter
"""
function SetUpProblemIC!(prob;  ρ = [],
                               ux = [], uy = [], uz =[],
                               bx = [], by = [], bz =[],
                               U₀x= [], U₀y= [], U₀z=[],
                               B₀x= [], B₀y= [], B₀z=[])
  sol = prob.sol;
  vars = prob.vars;
  grid = prob.grid;
  params = prob.params;
  if prob.flag.c
    if ρ == []
      error("User declare compressibility but no density IC was set.")
    else
      @views sol₀ =  sol[:, :, :, params.ρ_ind];
      copyto!(vars.ρ, ρ);
      mul!(sol₀ , grid.rfftplan, vars.ρ);
    end
  end
  if prob.dye.dyeflag
    if ρ ==[]
      warning("User declare the dye but no dye is set")
    else
      copyto!(prob.dye.ρ, ρ);
      mul!(prob.dye.tmp.sol₀, grid.rfftplan, prob.dye.ρ);
    end
  end

  # Copy the data to both output and solution array
  if (! prob.flag.e) 
    for (uᵢ,prob_uᵢ,uᵢind) in zip([ux,uy,uz],[vars.ux,vars.uy,vars.uz],
                                [params.ux_ind,params.uy_ind,params.uz_ind])
      if uᵢ != [] 
        @views sol₀ =  sol[:, :, :, uᵢind];
        copyto!(prob_uᵢ,uᵢ);
        if prob.flag.c 
          mul!(sol₀ , grid.rfftplan, @. vars.ρ*prob_uᵢ);
        else
          mul!(sol₀ , grid.rfftplan, prob_uᵢ);
        end
      end
    end
  end
  # copy b-field data
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
  # copy solid domin data
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
    DivFreeSpectraMap(Nx,Ny,Nz)

Construct a Div Free Spectra Vector Map with power-law relation
  Keyword arguments
=================
- `Nx/Ny/Nz`: size of the Vector Map
- `k0` : Slope of the Map 
- `b`  : Anisotropy of the Map
"""
function DivFreeSpectraMap( Nx::Int, Ny::Int, Nz::Int;
                            Lx = 2π,
                            dev = CPU(), 
                            P = 1, k0 = -5/3/2, b = 1, T = Float64, k_peak = 0.0)
  grid = ThreeDGrid(dev; nx = Nx, Lx = Lx, ny=Ny, nz=Nz, T = T,nthreads = 8);
  return DivFreeSpectraMap( grid; k_peak = k_peak, P = P, k0 = k0, b = b);
end

function DivFreeSpectraMap( grid;
                            k_peak = 0.0, P = 1, k0 = -5/3/2, b = 1)
    
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
  @. Fk[1,:,:] .= 0;
  Fk[k.<k_peak] .= 0;
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
function DivFreeSpectraMap2( Nx::Int, Ny::Int, Nz::Int;
                            Lx = 2π,
                            dev = CPU(), 
                            k0 = 1.0, σ² = 1.0,  b = 1, T = Float64, k_peak = 0.0)
  grid = ThreeDGrid(dev; nx = Nx, Lx = Lx, ny=Ny, nz=Nz, T = T,nthreads = 8);
  return DivFreeSpectraMap( grid; k_peak = k_peak, P = P, k0 = k0, b = b);
end

function DivFreeSpectraMap2( grid;
                             k_peak = 0.0, P = 1, k0 = 1.0, σ² = 1.0,  b = 1)
    
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
  CUDA.@allowscalar Fk[1,1,1] = 0.0;
  @. Fk[1,:,:] .= 0;
  Fk  = @. √(exp(-(k.-k0)^2/σ²)/2/π)*k⁻¹;
    
  e1x = @.  ky/k⊥;
  e1y = @. -kx/k⊥;
  e2x = @. kx*kz/k⊥*k⁻¹;
  e2y = @. ky*kz/k⊥*k⁻¹;
  e2z = @. -k⊥*k⁻¹;
  e1x[isnan.(e1x)] .= 0;
  e1y[isnan.(e1y)] .= 0;
  e2y[isnan.(e2y)] .= 0;
  e2x[isnan.(e2x)] .= 0;
    
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

"""
    readMHDFlows(FileName)

Function of reading the HDF5 file written by MHDFlows
  Keyword arguments
=================
- `FileName`: string of the file path location of the files
"""
function readMHDFlows(FileName)
    F32(A::Array) = convert(Array{Float32,3},A);
    f = h5open(FileName);
    iv = F32(read(f,"i_velocity"));
    jv = F32(read(f,"j_velocity"));
    kv = F32(read(f,"k_velocity"));
    ib = F32(read(f,"i_mag_field"));
    jb = F32(read(f,"j_mag_field"));
    kb = F32(read(f,"k_mag_field"));
    t  = read(f,"time");
    close(f)
    return iv,jv,kv,ib,jb,kb,t
end