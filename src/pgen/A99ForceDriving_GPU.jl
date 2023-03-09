# ----------
# Problem Generation Module: A99 Turbulence template For GPU only 
# ----------
module A99GPU
  using CUDA
  mutable struct A99_vars{T}
    A   :: T
    b   :: T
    σ²  :: T
    kf  :: T
  end

  function GetA99vars_And_function(::Dev, nx::Int,ny::Int,nz::Int; T = Float32) where Dev

    A  = convert(T,1.0);
    b  = convert(T,1.0);
    σ² = convert(T,1.0);
    kf = convert(T,1.0);
    A99 = A99_vars(A,b,σ²,kf)

    return  A99, A99ForceDriving!, SetUpFk!  
  end

  function SetUpFk!(prob; kf = 2.0, P = 1.0, σ= 1.0, b = 1.0)
    AT   = Array;
    grid = prob.grid;
    T    = eltype(grid)

    kx,ky,kz  = AT(grid.kr),AT(grid.l),AT(grid.m);
    Lx,Ly,Lz  = grid.Lx,grid.Ly,grid.Lz;
    dx,dy,dz  = grid.dx,grid.dy,grid.dz;
    
    k⁻¹  =  sqrt.(AT(grid.invKrsq));
    k    =  sqrt.(AT(grid.Krsq));
    k⊥   = @. √(kx^2 + ky^2);
    dk⁻² = @. 1/(k+1)^2;
    
    ∫Fkdk  = sum(@. exp(-(k.-kf)^2/σ^2)*dk⁻²)
    A      = sqrt(P*3*(Lx/dx)*(Ly/dy)*(Lz/dz)/∫Fkdk*(1/dx/dy/dz));
    
    prob.vars.usr_vars.A  = T(A  )
    prob.vars.usr_vars.σ² = T(σ^2)
    prob.vars.usr_vars.b  = T(b  )
    prob.vars.usr_vars.kf = T(b  )
    
    return nothing
  end

  function A99ForceDriving!(N, sol, t, clock, vars, params, grid)

    # A99 Force parameter
    T    = eltype(grid)
    A  = vars.usr_vars.A::T
    b  = vars.usr_vars.b::T
    kf = vars.usr_vars.kf::T
    σ² = vars.usr_vars.σ²::T
    
    kx,ky,kz = grid.kr, grid.l, grid.m
    #ky       = params.usr_params.ky
    # "pointer"
    ∂uxh∂t = view(N,:,:,:,params.ux_ind)
    ∂uyh∂t = view(N,:,:,:,params.uy_ind)
    ∂uzh∂t = view(N,:,:,:,params.uz_ind)
                 
    # Set up of CUDA threads & block
    threads = ( 32, 8, 1 ) #(9,9,9)
    blocks  = ( ceil(Int,size(N,1)/threads[1]), ceil(Int,size(N,2)/threads[2]), ceil(Int,size(N,3)/threads[3]) )  
    
    @cuda blocks = blocks threads = threads A99Force_Driving_CUDA!(∂uxh∂t, ∂uyh∂t, ∂uzh∂t, kx, ky, kz, 
                                                                   A, kf, σ², b)

    return nothing
  end


  function A99Force_Driving_CUDA!(∂uxh∂t,∂uyh∂t,∂uzh∂t,kx_,ky_,kz_,A,kf,σ²,b)
    #define the i,j,k
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    z = (blockIdx().z - 1) * blockDim().z + threadIdx().z 
    
    nx,ny,nz = size(∂uxh∂t)
    if z ∈ (1:nz) && y ∈ (1:ny) && x ∈ (1:nx)
      # Reason : https://github.com/FourierFlows/Fou/rierFlows.jl/issues/326 
      if size(y,1) > 1
        kx,ky,kz  = kx_[x], ky_[x,y], kz_[z]
      else
        @inbounds kx,ky,kz  = kx_[x], ky_[y], kz_[z]
      end
      k    =  √(kx^2 + ky^2 + kz^2)
      k⊥   =  √(kx^2 + kz^2)
      k⁻¹  =  k > 0.0 ? 1/k : 0.0
      Fk   =  A*√(exp(-(k-kf)^2/σ²)/2/π)*k⁻¹
      
      #e1y = k⊥ <= 0.0 ?  0.0 : -kx/k⊥;
      #e2y = k⊥ <= 0.0 ?  0.0 :  ky*kz/k⊥*k⁻¹
      #e2z = -k⊥*k⁻¹  
      e1x = k⊥ <= 0.0 ?  0.0 :  kz/k⊥      
      e1z = k⊥ <= 0.0 ?  0.0 : -kx/k⊥;   

      e2x = k⊥ <= 0.0 ?  0.0 :  kx*ky/k⊥*k⁻¹        
      e2y = -k⊥*k⁻¹                                 
      e2z = k⊥ <= 0.0 ?  0.0 :  kz*ky/k⊥*k⁻¹       

      eⁱᶿ = exp(rand()*2π*im)
      Φ   = rand()*π
      gi  = -tanh(b*(Φ - π/2))/tanh(b*π/2)
      gi  = abs(gi) >= 1.0 ? sign(gi)*1.0 : gi

      @inbounds ∂uxh∂t[x,y,z] += A*Fk*eⁱᶿ*gi*e1x
      @inbounds ∂uzh∂t[x,y,z] += A*Fk*eⁱᶿ*gi*e1z
        
      eⁱᶿ = exp(rand()*2π*im)
      gj = √(1 - gi^2)
          
      @inbounds ∂uxh∂t[x,y,z] += A*Fk*eⁱᶿ*gj*e2x
      @inbounds ∂uyh∂t[x,y,z] += A*Fk*eⁱᶿ*gj*e2y
      @inbounds ∂uzh∂t[x,y,z] += A*Fk*eⁱᶿ*gj*e2z

      if x == 1 || x == nx # the indexing need to change when applying CuFFTMp
        # for rfft, the complex X[1] == X[N/2+1] == 0
        # Reason : https://github.com/FourierFlows/Fou/rierFlows.jl/issues/326 
        @inbounds ∂uxh∂t[x,y,z] = real(∂uxh∂t[x,y,z])
        @inbounds ∂uyh∂t[x,y,z] = real(∂uyh∂t[x,y,z])
        @inbounds ∂uzh∂t[x,y,z] = real(∂uzh∂t[x,y,z])
      end
    end

    return nothing 
  end

end