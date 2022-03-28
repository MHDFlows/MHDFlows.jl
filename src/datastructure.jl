module datastructure
	#Setting up the data structure

using 
  CUDA,
  Statistics,
  SpecialFunctions,
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

export SetMHDVars,
       SetHDVars,
       MHDParams,
       HDParams

# ----------
# Parameters and data structure for HD and MHD problem
# ----------


abstract type MHDVars <: AbstractVars end
abstract type HDVars <: AbstractVars end

struct MVars{Aphys, Atrans} <: MHDVars
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
    "Non-linear term 2"
     nonlin2 :: Aphys
    "Fourier transform of Non-linear term"
    nonlinh2 :: Atrans
    "Non-linear term 3"
     nonlin3 :: Aphys
    "Fourier transform of Non-linear term"
    nonlinh3 :: Atrans

end

struct MHDParams{T} <: AbstractParams
    
  "small-scale (hyper)-viscosity coefficient for v"
    ν :: T
  "small-scale (hyper)-viscosity coefficient for b"
    η :: T
  "(hyper)-viscosity order, `nν```≥ 1``"
    nν :: Int

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
    
end

struct HVars{Aphys, Atrans} <: MHDVars
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
  "Non-linear term 2"
    nonlin2 :: Aphys
  "Fourier transform of Non-linear term"
    nonlinh2 :: Atrans
  "Non-linear term 3"
    nonlin3 :: Aphys
  "Fourier transform of Non-linear term"
    nonlinh3 :: Atrans

end

struct HDParams{T} <: AbstractParams
    
  "small-scale (hyper)-viscosity coefficient for v"
   ν :: T
  "(hyper)-viscosity order, `nν```≥ 1``"
   nν :: Int

  "Array Indexing for velocity"
   ux_ind :: Int
   uy_ind :: Int
   uz_ind :: Int
    
end


function SetMHDVars(::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ux  uy  uz  bx  by bz nonlin1 nonlin2 nonlin3
  @devzeros Dev Complex{T} (grid.nkr, grid.nm, grid.nl) uxh uyh uzh bxh byh bzh nonlinh1 nonlinh2 nonlinh3
  
  return MVars( ux,  uy,  uz,  bx,  by,  bz,
              uxh, uyh, uzh, bxh, byh, bzh,
              nonlin1, nonlinh1, nonlin2, nonlinh2, nonlin3, nonlinh3);
end

function SetHDVars(::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ux  uy  uz nonlin1 nonlin2 nonlin3
  @devzeros Dev Complex{T} (grid.nkr, grid.nm, grid.nl) uxh uyh uzh nonlinh1 nonlinh2 nonlinh3
  
  return HVars( ux,  uy,  uz,  uxh, uyh, uzh,
              nonlin1, nonlinh1, nonlin2, nonlinh2, nonlin3, nonlinh3);
end



end