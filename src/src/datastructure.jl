# ----------
# Parameters and data structure for HD and MHD problem
# ----------

struct MVars{Aphys, Atrans, usr_var} <: MHDVars
    "x-component of velocity"
        ux :: Aphys
    "y-component of velocity"
        uy :: Aphys
    "z-component of velocity"
        uz :: Aphys
    "x-component of B-field"
        bx :: Aphys
    "y-component of B-field"
        by :: Aphys
    "z-component of B-field"
        bz :: Aphys
    "Fourier transform of x-component of velocity"
       uxh :: Atrans
    "Fourier transform of y-component of velocity"
       uyh :: Atrans
    "Fourier transform of z-component of velocity"
       uzh :: Atrans
    "Fourier transform of x-component of B-field"
       bxh :: Atrans
    "Fourier transform of y-component of B-field"
       byh :: Atrans
    "Fourier transform of z-component of B-field"
       bzh :: Atrans

    # Temperatory Cache 
    "Non-linear term 1"
     nonlin1 :: Aphys
    "Fourier transform of Non-linear term"
    nonlinh1 :: Atrans

    # User Defined Vars
    "User Defined Vars"
    usr_vars :: usr_var
end

struct HVars{Aphys, Atrans, usr_var} <: MHDVars
  "x-component of velocity"
    ux :: Aphys
  "y-component of velocity"
    uy :: Aphys
  "z-component of velocity"
    uz :: Aphys
  "Fourier transform of x-component of velocity"
   uxh :: Atrans
  "Fourier transform of y-component of velocity"
   uyh :: Atrans
  "Fourier transform of z-component of velocity"
   uzh :: Atrans
  # Temperatory Cache 
  "Non-linear term 1"
   nonlin1 :: Aphys
  "Fourier transform of Non-linear term"
   nonlinh1 :: Atrans

  # User Defined Vars
  "User Defined Vars"
    usr_vars :: usr_var
end

struct MHDParams_VP{Aphys,usr_param} <: AbstractParams
    
  "small-scale (hyper)-viscosity coefficient for v"
    ν :: Number
  "small-scale (hyper)-viscosity coefficient for b"
    η :: Number
  "(hyper)-viscosity order, `nν```≥ 1``"
    nν :: Int
  "(hyper)-resisivity order, `nη```≥ 1``"
    nη :: Int

  "Array Indexing for velocity"
    ux_ind :: Int
    uy_ind :: Int
    uz_ind :: Int
    
  "Array Indexing for B-field"
    bx_ind :: Int
    by_ind :: Int
    bz_ind :: Int

  "function that calculates the Fourier transform of the forcing, ``F̂``"
    calcF! :: Function 
  
  "Volume penzlization method paramter"
    χ   :: Aphys
    U₀x :: Aphys
    U₀y :: Aphys
    U₀z :: Aphys
    B₀x :: Aphys
    B₀y :: Aphys
    B₀z :: Aphys

  "User defined params"
  usr_params :: usr_param

end

struct HDParams_VP{Aphys,usr_param} <: AbstractParams
    
  "small-scale (hyper)-viscosity coefficient for v"
    ν :: Number
  "(hyper)-viscosity order, `nν```≥ 1``"
    nν :: Int

  "Array Indexing for velocity"
    ux_ind :: Int
    uy_ind :: Int
    uz_ind :: Int
    
  "function that calculates the Fourier transform of the forcing, ``F̂``"
    calcF! :: Function 
  
  "Volume penzlization method paramter"
    χ   :: Aphys
    U₀x :: Aphys
    U₀y :: Aphys
    U₀z :: Aphys

  "User defined params"
  usr_params :: usr_param   

end

struct HDParams{usr_param} <: AbstractParams
    
  "small-scale (hyper)-viscosity coefficient for v"
   ν :: Number
  "(hyper)-viscosity order, `nν```≥ 1``"
   nν :: Int

  "Array Indexing for velocity"
   ux_ind :: Int
   uy_ind :: Int
   uz_ind :: Int

  "function that calculates the Fourier transform of the forcing, ``F̂``"
   calcF! :: Function

  "User defined params"
 usr_params :: usr_param
end

struct MHDParams{usr_param} <: AbstractParams
    
  "small-scale (hyper)-viscosity coefficient for v"
    ν :: Number
  "small-scale (hyper)-viscosity coefficient for b"
    η :: Number
  "(hyper)-viscosity order, `nν```≥ 1``"
    nν :: Int
  "(hyper)-resisivity order, `nη```≥ 1``"
    nη :: Int
    
  "Array Indexing for velocity"
    ux_ind :: Int
    uy_ind :: Int
    uz_ind :: Int
    
  "Array Indexing for B-field"
    bx_ind :: Int
    by_ind :: Int
    bz_ind :: Int
  "function that calculates the Fourier transform of the forcing, ``F̂``"
    calcF! :: Function 
  
  "User defined params"
 usr_params :: usr_param

end


function SetMHDVars(::Dev, grid::AbstractGrid, usr_vars) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ux  uy  uz  bx  by bz nonlin1
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm) uxh uyh uzh bxh byh bzh nonlinh1
  
  return MVars( ux,  uy,  uz,  bx,  by,  bz,
              uxh, uyh, uzh, bxh, byh, bzh,
              nonlin1, nonlinh1, usr_vars);
end

function SetHDVars(::Dev, grid::AbstractGrid, usr_vars) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ux  uy  uz nonlin1
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm) uxh uyh uzh nonlinh1
  
  return HVars( ux,  uy,  uz,  uxh, uyh, uzh,
              nonlin1, nonlinh1, usr_vars);
end


# Functions of setting up the Vars and Params struct
function SetVars(dev, grid, usr_vars; B = false, VP = false)
  setvars = ifelse(B,SetMHDVars,SetHDVars);
  return setvars(dev, grid, usr_vars);
end

 function SetParams(::Dev, grid::AbstractGrid, calcF::Function, usr_params;
                     B = false, VP = false, ν = 0, η = 0, nν = 0, nη = 0) where Dev
  T = eltype(grid);
  usr_param = typeof(usr_params);
  if (B)
    if (VP)
      @devzeros Dev T (grid.nx, grid.ny, grid.nz) χ U₀x U₀y U₀z B₀x B₀y B₀z
      params = MHDParams_VP(ν, η, nν, nη, 1, 2, 3, 4, 5, 6, calcF, χ, U₀x, U₀y, U₀z, B₀x, B₀y, B₀z, usr_params)
    else
      params = MHDParams(ν, η, nν, nη, 1, 2, 3, 4, 5, 6, calcF, usr_params);
    end
  else
    if (VP)
      @devzeros Dev T (grid.nx, grid.ny, grid.nz) χ U₀x U₀y U₀z
      params = HDParams_VP(ν, nν, 1, 2, 3, calcF, χ, U₀x, U₀y, U₀z, usr_params);
    else
      params = HDParams(ν, nν, 1, 2, 3, calcF, usr_params);
    end
  end

  return params

end
