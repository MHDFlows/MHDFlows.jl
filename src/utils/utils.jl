function Getk²_and_fftplan(nx = 64, Lx = 2π, ny = nx, Ly = Lx, nz = nx, Lz = Lx;
                           nthreads=Sys.CPU_THREADS, effort=FFTW.MEASURE,
                           T=Float64, ArrayType=Array)

  nk = nx
  nl = ny
  nm = nz
  nkr = Int(nx/2 + 1)  
  # Wavenubmer
   k = ArrayType{T}(reshape( fftfreq(nx, 2π/Lx*nx), (nk, 1, 1)))
   l = ArrayType{T}(reshape( fftfreq(ny, 2π/Ly*ny), (1, nl, 1)))
   m = ArrayType{T}(reshape( fftfreq(nz, 2π/Lz*nz), (1, 1, nm)))
  kr = ArrayType{T}(reshape(rfftfreq(nx, 2π/Lx*nx), (nkr, 1, 1)))
  k² = @. k^2 + l^2 + m^2;
    
  FFTW.set_num_threads(nthreads);
  rfftplan = plan_rfft(ArrayType{T, 3}(undef, nx, ny, nz))
  
    return k², rfftplan;
end
