mutable struct SimpleGrid{i,i²,plan}
    k   :: i
    l   :: i
    m   :: i
    kr  :: i
    Ksq :: i²
    invKsq  :: i²
    Krsq    :: i²
    invKrsq :: i²
    rfftplan :: plan
end
Base.eltype(grid::SimpleGrid) = eltype(grid.k);

function GetSimpleThreeDGrid(nx = 64, Lx = 2π, ny = nx, Ly = Lx, nz = nx, Lz = Lx;
                             nthreads=Threads.nthreads(), effort=FFTW.MEASURE,
                             T=Float64, ArrayType=Array,dev=CPU())
  nk = nx
  nl = ny
  nm = nz
  nkr = Int(nx/2 + 1)  
  # Wavenubmer
  k = ArrayType{T}(reshape( fftfreq(nx, 2π/Lx*nx), (nk, 1, 1)))
  l = ArrayType{T}(reshape( fftfreq(ny, 2π/Ly*ny), (1, nl, 1)))
  m = ArrayType{T}(reshape( fftfreq(nz, 2π/Lz*nz), (1, 1, nm)))
  kr = ArrayType{T}(reshape(rfftfreq(nx, 2π/Lx*nx), (nkr, 1, 1)))

     Ksq = @. k^2 + l^2 + m^2
  invKsq = @. 1 / Ksq
  CUDA.@allowscalar  invKsq[1, 1, 1] = 0;

     Krsq = @. kr^2 + l^2 + m^2
  invKrsq = @. 1 / Krsq
  CUDA.@allowscalar invKrsq[1, 1, 1] = 0;
    
  FFTW.set_num_threads(nthreads);
  rfftplan = plan_rfft(ArrayType{T, 3}(undef, nx, ny, nz))
  
  return SimpleGrid(k,l,m,kr,Ksq, invKsq, Krsq, invKrsq, rfftplan);
end

#============================Shearing Grid=====================================================#
function GetShearingThreeDGrid(dev::Device=CPU(); nx, Lx, ny=nx, Ly=Lx, nz=nx, Lz=Lx,
                               x0=-Lx/2, y0=-Ly/2, z0=-Lz/2,
                               nthreads=Sys.CPU_THREADS, effort=FFTW.MEASURE, T=Float64,
                               aliased_fraction=1/3)
  device_array = FourierFlows.device_array

  dx = Lx/nx
  dy = Ly/ny
  dz = Lz/nz

  nk = nx
  nl = ny
  nm = nz
  nkr = Int(nx/2 + 1)

  # Physical grid
  x = range(T(x0), step=T(dx), length=nx)
  y = range(T(y0), step=T(dy), length=ny)
  z = range(T(z0), step=T(dz), length=nz)

  # Wavenubmer grid
   k = device_array(dev){T}(reshape( fftfreq(nx, 2π/Lx*nx), (nk, 1, 1)))
 l1D = device_array(dev){T}(reshape( fftfreq(ny, 2π/Ly*ny), (1, nl, 1)))
 l2D = device_array(dev){T}(reshape( fftfreq(ny, 2π/Ly*ny), (1, nl, 1)) .* ones(nkr, 1, 1))
   m = device_array(dev){T}(reshape( fftfreq(nz, 2π/Lz*nz), ( 1, 1, nm)))
  kr = device_array(dev){T}(reshape(rfftfreq(nx, 2π/Lx*nx), (nkr, 1, 1)))

     Ksq = @. k^2 + l1D^2 + m^2
  invKsq = @. 1 / Ksq
  CUDA.@allowscalar invKsq[1, 1, 1] = 0

     Krsq = @. kr^2 + l1D^2 + m^2
  invKrsq = @. 1 / Krsq
  CUDA.@allowscalar invKrsq[1, 1, 1] = 0

  # FFT plans
  FFTW.set_num_threads(nthreads)
  fftplan  = FourierFlows.plan_flows_fft(device_array(dev){Complex{T}, 3}(undef, nx, ny, nz), flags=effort)
  rfftplan = FourierFlows.plan_flows_rfft(device_array(dev){T, 3}(undef, nx, ny, nz), flags=effort)

  kalias, kralias = FourierFlows.getaliasedwavenumbers(nk, nkr,         aliased_fraction)
  lalias, _       = FourierFlows.getaliasedwavenumbers(nl, Int(nl/2+1), aliased_fraction)
  malias, _       = FourierFlows.getaliasedwavenumbers(nm, Int(nm/2+1), aliased_fraction)
  
  R = typeof(x)
  A = typeof(k)
  Axy = typeof(l2D)
  Tfft = typeof(fftplan)
  Trfft = typeof(rfftplan)
  Talias = typeof(kalias)
  D = typeof(dev)

  return ShearingThreeDGrid{T, A, Axy, R, Tfft, Trfft, Talias, D}(dev, nx, ny, nz, nk, nl, nm, nkr,
                                                                  dx, dy, dz, Lx, Ly, Lz, x, y, z, k, l1D, l2D, m, kr,
                                                                  Ksq, invKsq, Krsq, invKrsq, fftplan, rfftplan, 
                                                                  aliased_fraction, kalias, kralias, lalias, malias)
end

struct ShearingThreeDGrid{T<:AbstractFloat, Tk, Tky, Tx, Tfft, Trfft, Talias, D}  <: FourierFlows.AbstractGrid{T, Tk, Talias, D}
    "device which the grid lives on"
           device :: D
    "number of points in ``x``"
               nx :: Int
    "number of points in ``y``"
               ny :: Int
    "number of points in ``z``"
               nz :: Int
    "number of wavenumbers in ``x``"
               nk :: Int
    "number of wavenumbers in ``y``"
               nl :: Int
    "number of wavenumbers in ``z``"
               nm :: Int
    "number of positive wavenumers in ``x`` (real Fourier transforms)"
               nkr :: Int
    "grid spacing in ``x``"
                dx :: T
    "grid spacing in ``y``"
                dy :: T
    "grid spacing in ``z``"
                dz :: T
    "domain extent in ``x``"
                Lx :: T
    "domain extent in ``y``"
                Ly :: T
    "domain extent in ``z``"
                Lz :: T
    "range with ``x``-grid-points"
                 x :: Tx
    "range with ``y``-grid-points"
                 y :: Tx
    "range with ``z``-grid-points"
                 z :: Tx
    "array with ``x``-wavenumbers"
                 k :: Tk
    "array with ``y``-wavenumbers(1D)"
               l1D :: Tk
    "array with ``y``-wavenumbers(2D)"
                 l :: Tky
    "array with ``z``-wavenumbers"
                 m :: Tk
    "array with positive ``x``-wavenumbers (real Fourier transforms)"
                kr :: Tk
    "array with squared total wavenumbers, ``k² + l² + m²``"
               Ksq :: Tk
    "array with inverse squared total wavenumbers, ``1 / (k² + l² + m²)``"
            invKsq :: Tk
    "array with squared total wavenumbers for real Fourier transforms, ``kᵣ² + l² + m²``"
              Krsq :: Tk
    "array with inverse squared total wavenumbers for real Fourier transforms, ``1 / (kᵣ² + l² + m²)``"
           invKrsq :: Tk
    "the FFT plan for complex-valued fields"
           fftplan :: Tfft
    "the FFT plan for real-valued fields"
          rfftplan :: Trfft
    "the fraction of wavenumbers that are aliased (e.g., 1/3 for quadradic nonlinearities)"
  aliased_fraction :: T
    "range of the indices of aliased ``x``-wavenumbers"
            kalias :: Talias
    "range of the indices of aliased positive ``x``-wavenumbers (real Fourier transforms)"
           kralias :: Talias
    "range of the indices of aliased ``y``-wavenumbers"
            lalias :: Talias
    "range of the indices of aliased ``m``-wavenumbers"
            malias :: Talias
end


Base.eltype(grid::ShearingThreeDGrid) = eltype(grid.x)
#===============================================================================================#
function Move_Data_to_Prob!(data,real,sol,grid)
  copyto!(real, deepcopy(data));
  mul!(sol, grid.rfftplan, real);   
  return nothing
end