struct MHDParams_VP{Aphys,usr_param,to} <: AbstractParams
    
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

  "Debug timer"
  debugTimer :: to

end

struct MHDParams{usr_param,to} <: AbstractParams
    
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

 "Debug timer"
  debugTimer :: to

end

struct EMHDParams{usr_param,to} <: AbstractParams
    
  "small-scale (hyper)-viscosity coefficient for b"
    η :: Number

  "(hyper)-resisivity order, `nη```≥ 1``"
    nη :: Int
    
  "Array Indexing for B-field"
    bx_ind :: Int
    by_ind :: Int
    bz_ind :: Int
  "function that calculates the Fourier transform of the forcing, ``F̂``"
    calcF! :: Function 
  
  "User defined params"
 usr_params :: usr_param

 "Debug timer"
  debugTimer :: to

end

struct CMHDParams{usr_param,to} <: AbstractParams
  "speed of sound"
        cₛ :: Number
  "small-scale (hyper)-viscosity coefficient for v"
        ν :: Number
  "small-scale (hyper)-viscosity coefficient for b"
        η :: Number
  "(hyper)-viscosity order, `nν```≥ 1``"
        nν :: Int
  "(hyper)-resisivity order, `nη```≥ 1``"
        nη :: Int
    
  "Array Indexing for density/velocity"
     ρ_ind :: Int
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

 "Debug timer"
  debugTimer :: to

end
