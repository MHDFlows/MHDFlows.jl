# ----------
# Vector Calculus Module, Only work on peroideric boundary!
# ----------

function Curl(B1::Array,B2::Array,B3::Array;
              Lx = 2π, Ly = Lx, Lz = Lx,T = Float32)
    # Wrapper for Curl Function
    nx,ny,nz = size(B1);
    grid = GetSimpleThreeDGrid(nx, Lx, ny, Ly, nz, Lz, T = T);
    cB1,cB2,cB3 = Curl(B1,B2,B3,grid)
    return cB1,cB2,cB3;
end

function Curl(B1,B2,B3,grid)
    #funtion of computing ∇×Vector using the fourier method
    # fft(∇×Vector) -> im * k × V
    #| i j k  |
    #| x y z  |
    #|B1 B2 B3|
    #
    nx,ny,nz = size(B1);
    dev = typeof(B1) <: Array ? CPU() : GPU();
    T   = eltype(grid);

    @devzeros typeof(dev) Complex{T} (div(nx,2)+1,ny,nz) B1h B2h B3h CB1h CB2h CB3h
    @devzeros typeof(dev)         T  (         nx,ny,nz) cB1 cB2 cB3

    mul!(B1h, grid.rfftplan, B1); 
    mul!(B2h, grid.rfftplan, B2); 
    mul!(B3h, grid.rfftplan, B3);
    
    kx,ky,kz = grid.kr,grid.l,grid.m; 
    @. CB1h = im*(ky*B3h - kz*B2h);
    @. CB2h = im*(kz*B1h - kx*B3h);
    @. CB3h = im*(kx*B2h - ky*B1h);
    
    ldiv!(cB1, grid.rfftplan, CB1h);  
    ldiv!(cB2, grid.rfftplan, CB2h);
    ldiv!(cB3, grid.rfftplan, CB3h);
    return cB1,cB2,cB3;
end

function Div(B1::Array,B2::Array,B3::Array;
             Lx = 2π, Ly = Lx, Lz = Lx,T = Float32)
    nx,ny,nz = size(B1);
    grid = GetSimpleThreeDGrid(nx, Lx, ny, Ly, nz, Lz, T = T);
    cB1 = Div(B1,B2,B3,grid);

    return cB1
end

function Div(B1,B2,B3,grid)
    #funtion of computing ∇̇ ⋅ Vector using the fourier method
    # fft(∇·Vector) -> im * k ⋅ V
    # = im* (x*B1 + y*B2 + z*B3)

    nx,ny,nz = size(B1);
    dev = typeof(B1) <: Array ? CPU() : GPU();
    T   = eltype(grid);

    @devzeros typeof(dev) Complex{T} (div(nx,2)+1,ny,nz) B1h B2h B3h Dot
    @devzeros typeof(dev)         T  (         nx,ny,nz) cB1
    
    mul!(B1h, grid.rfftplan, B1); 
    mul!(B2h, grid.rfftplan, B2); 
    mul!(B3h, grid.rfftplan, B3);

    kx,ky,kz = grid.kr,grid.l,grid.m; 
    @. Dot = im*(kx*B1h+ky*B2h+kz*B3h)
    
    ldiv!(cB1, grid.rfftplan, Dot);  

    return cB1
end

function ∂i(B1::Array, direction; Lx = 2π, Ly = Lx, Lz = Lx,T = eltype(B1))
  nx,ny,nz = size(B1);
  dev = typeof(B1) <: Array ? CPU() : GPU();
  grid = GetSimpleThreeDGrid(nx, Lx, ny, Ly, nz, Lz, T = T);
  return ∂i(B1,grid, direction);
end

function ∂i(B1::Array,grid, direction)
    # funtion of computing x/y/z-direction of ∇̇ ⋅ Vector using the fourier method
    # fft(∂_i(Vector)) -> im * k_i ⋅ V

    nx,ny,nz = size(B1);
    dev = typeof(B1) <: Array ? CPU() : GPU();
    T   = eltype(grid);

    @devzeros typeof(dev) Complex{T} (div(nx,2)+1,ny,nz) B1h
    @devzeros typeof(dev)         T  (         nx,ny,nz) cB1
    
    mul!(B1h, grid.rfftplan, B1);
    kx,ky,kz = grid.kr,grid.l,grid.m;
    if  direction == "x"
      @. B1h = im*kx*B1h
    elseif direction == "y"
      @. B1h = im*ky*B1h
    elseif direction =="z"
      @. B1h = im*kz*B1h
    else
      error("Wrong driection declared")
    end
    ldiv!(cB1, grid.rfftplan, B1h); 

    return cB1
end


∇X(A1,A2,A3;Lx = 2π, Ly = Lx, Lz = Lx,T = eltype(A1)) =  Curl(A1,A2,A3;Lx = Lx, Ly = Ly, Lz = Lz,T = eltype(A1));
#∇·(A1,A2,A3;Lx = 2π, Ly = Lx, Lz = Lx,T = eltype(A1)) =   Div(A1,A2,A3;Lx = Lx, Ly = Ly, Lz = Lz,T = eltype(A1));


function LaplaceSolver(B; Lx=2π, Ly = Lx, Lz = Lx, T = Float32)
    nx,ny,nz = size(B);
    grid = GetSimpleThreeDGrid(nx, Lx, ny, Ly, nz, Lz, T = T);
    Φ   = LaplaceSolver(B,grid);
    return Φ
end

function LaplaceSolver(B,grid)
    #=
    funtion of computing ΔΦ = B using the fourier method, must be peroidic condition
    Considering in k-space, k² Φ' = B', we would get Φ = F(B'/k²)
    =#
    k⁻² = grid.invKrsq;
    T   = eltype(grid);
    nx,ny,nz = size(B);
    Φ    = zeros(T,nx,ny,nz);
    Bh   = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    mul!(Bh, grid.rfftplan, B); 
    @. Bh*=k⁻²;
    ldiv!(Φ, grid.rfftplan, deepcopy(Bh));
    return Φ;
end

function Crossproduct(A1,A2,A3,B1,B2,B3)
  C1 = @.  (A2*B3 - A3*B2);
  C2 = @. -(A1*B3 - A3*B1); 
  C3 = @.  (A1*B2 - A2*B1);
  return C1,C2,C3
end

#Define the synatex sugar for crossproduct and dot product
(a1,a2,a3)×(b1,b2,b3) = Crossproduct(a1,a2,a3,b1,b2,b3);
(a1,a2,a3)⋅(b1,b2,b3) = Dotproduct(a1,a2,a3,b1,b2,b3); 

function Dotproduct(A1,A2,A3,B1,B2,B3)
    return A1.*B1 + A2.*B2 + A3.*B3 
end

function DivVCorrection!(ux,uy,uz,grid)
#= 
   Possion Solver for periodic boundary condition
   As in VP method, ∇ ⋅ V = 0 may not hold, V = ∇×Ψ + ∇Φ -> ∇ ⋅ V = ∇² Φ
   We need to find Φ and remove it using a Poission Solver 
   Here we are using the Fourier Method to find the Φ
   In Real Space,  
   ∇² Φ = ∇ ⋅ V   
   In k-Space,  
   ∑ᵢ -(kᵢ)² Φₖ = i∑ᵢ kᵢ(Vₖ)ᵢ
   Φₖ = i∑ᵢ kᵢ(Vₖ)ᵢ/k²
   Vⱼ_new = Vₖⱼ + kⱼ i∑ᵢ kᵢ(Vₖ)ᵢ/k²; 
=#  

  T = eltype(grid);  
  nx,ny,nz = grid.nx,grid.ny,grid.nz;  
  uxh = zeros(Complex{T},(div(nx,2)+1,ny,nz));
  uyh = zeros(Complex{T},(div(nx,2)+1,ny,nz));
  uzh = zeros(Complex{T},(div(nx,2)+1,ny,nz));
  mul!(uxh, grid.rfftplan, ux); 
  mul!(uyh, grid.rfftplan, uy); 
  mul!(uzh, grid.rfftplan, uz);

  #find Φₖ
  kᵢ,kⱼ,kₖ = grid.kr,grid.l,grid.m;
  k⁻² = grid.invKrsq;
  ∑ᵢkᵢUᵢh_k² = 0 .*copy(uxh);
  ∑ᵢkᵢUᵢ_k²  = 0 .*copy(ux);  
    
  ∑ᵢkᵢUᵢh_k² = @. im*(kᵢ*uxh + kⱼ*uyh + kₖ*uzh);
  ∑ᵢkᵢUᵢh_k² = @. -∑ᵢkᵢUᵢh_k²*k⁻²;  # Φₖ
  
  # B  = B* - ∇Φ = Bᵢ - kᵢΦₖ  
  uxh  .-= kᵢ.*∑ᵢkᵢUᵢh_k²;
  uyh  .-= kⱼ.*∑ᵢkᵢUᵢh_k²;
  uzh  .-= kₖ.*∑ᵢkᵢUᵢh_k²;
  
  #Update to Real Space vars
  ldiv!(ux, grid.rfftplan, deepcopy(uxh));# deepcopy() since inverse real-fft destroys its input
  ldiv!(uy, grid.rfftplan, deepcopy(uyh));# deepcopy() since inverse real-fft destroys its input
  ldiv!(uz, grid.rfftplan, deepcopy(uzh));# deepcopy() since inverse real-fft destroys its input
  return ux,uy,uz
end
