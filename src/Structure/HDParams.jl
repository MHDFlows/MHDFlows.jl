struct HDParams_VP{Aphys,usr_param,to} <: AbstractParams
    
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

  "Debug timer"
  debugTimer :: to

end

struct HDParams{usr_param,to} <: AbstractParams
    
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

 "Debug timer"
  debugTimer :: to

end

struct CHDParams{usr_param,to} <: AbstractParams
  "speed of sound"
        cₛ :: Number
  "small-scale (hyper)-viscosity coefficient for v"
        ν :: Number
  "(hyper)-viscosity order, `nν```≥ 1``"
       nν :: Int

  "Array Indexing for density/velocity"
    ρ_ind :: Int
   ux_ind :: Int
   uy_ind :: Int
   uz_ind :: Int

  "function that calculates the Fourier transform of the forcing, ``F̂``"
   calcF! :: Function

  "User defined params"
 usr_params :: usr_param

 "Debug timer"
  debugTimer :: to

end
