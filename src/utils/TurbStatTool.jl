module TurbStatTool
# Turbulence Statistics Tool 
# Provding 2D/3D Sturcture Function/ Correrlation Function 
# Author : Ka Wai HO @ UW-Madison
# Date   : 6 Jan 2023
using CUDA
using PyPlot, FFTW, Statistics, LsqFit
using MuladdMacro:@muladd
using ProgressMeter

# Structure function/ Correrlation Function
export SFr,CF,SFC

# contour/ fitting function
export fitline, Getcontour

#================ synatex sugar & basic function/type definition ======================# 

# Type definition from LazType
const  Mat{T}  = Array{T,2};
const Cube{T}  = Array{T,3};

# dot product
(a1,a2)⋅(b1,b2) = @. a1*b1 + a2*b2 

# cross product
#==
| x̂  ŷ  ẑ|
|a1 a2 a3|
|b1 b2 b3|
==#
(a1,a2,a3)×(b1,b2,b3) = ( @. a2*b3-a3*b2, @. a3*b1-a1*b3, @. a1*b2-a2*b1)

s(vx,vy,vz) = (vx,vy,vz)⋅(vx,vy,vz)
s(vx,vy)    = (vx,vy)⋅(vx,vy)

function t(a1,a2,a3,b1,b2,b3)
  x=(a1,a2,a3)⋅(b1,b2,b3)/√(s(a1,a2,a3)*s(b1,b2,b3));
  if (x>1) 
  	x=1 
  end;
  if (x<-1)
  	x=-1
  end;
  return acos(x)
end

function t(a1,a2,b1,b2)
  x = (a1,a2)⋅(b1,b2)/√(s(a1,a2)*s(b1,b2));
  if (x>1) 
  	x=1 
  end;
  if (x<-1)
  	x=-1
  end;
  return acos(x)
end

RoundUpInt(x::Number) = round(Int,x,RoundUp)
RoundInt(x::Number)   = round(Int,x) 
getind(kll::Number,kpp::Number,Rx::Number,Ry::Number) = (RoundInt(kll),RoundInt(kpp))

#================================== Main Function =========================================#

#Auto-Correlation function for 2D/3D **periodic** Map/Cube
CF(V::Cube) = fftshift((real(ifft(abs.(fft(V)).^2))));

#provding 2D/3D 2 point structure point function for scalar 
#functional form: SF(R) = <|V(x+R)-V(x)|^2>
#Note : R is 3D vector 
SFC(V) = 2*(mean(V).-CF(V));

# provding 3D B-field 2 point structure function for vector (iv,jv,kv) 
# Global Mode For strong B-field case
# Local Mode For weak B-field case
function SFr(iv,jv,kv, ib,jb,kb, mode;GPU = true, r=100,Nseed=500)
  print("mode = $(mode), GPU acceleration = $(GPU)")
  if mode == "Global"
  	SFVr = SFr_Global(iv,jv,kv,ib,jb,kb)
  elseif mode == "Local" &&  GPU
  	SFVr = SFr_local_GPU(iv,jv,kv,ib,jb,kb; Nseed = Nseed)
  elseif mode == "Local" && !GPU
  	SFVr = SFr_local_GPU(iv,jv,kv,ib,jb,kb; r = r, Nseed = Nseed)
  end
  return SFVr
end


# 2 point radial structure function for 3D peroderic simulation
function SF₂1D(Vx::Cube,Vz::Cube,Vy::Cube)
  Nx,Ny,Nz = size(Vx);
  R        = sqrt(s(div(Nx,2),div(Ny,2),div(Nz,2)));
  
  #get the structure function 
  SFVx  = SFC(Vx);
  SFVy  = SFC(Vy);
  SFVz  = SFC(Vz);
  #get the vector structure function
  SFV   = @. SFVx + SFVy + SFVz;
  #declaring the output
  R    = RoundUpInt(R);
  Mask = zeros(2*R);
  SFVr = zeros(2*R);

  for k in 1:Nz, j in 1:Ny, i in 1:Nx
    # get the k vector
    idx = i-div(Nx,2);
    jdx = j-div(Ny,2);
    kdx = k-div(Nz,2);
    kk  = round(Int,sqrt(s(idx,jdx,kdx)))
    if kk>0
      Mask[kk] += 1; 
      SFVr[kk] += SFV[i,j,k];
    end
  end
  SFVr./=Mask;
  SFVr
end

#provding 2 point Global structure point function for vector (Vx,Vy,Vz)
#Note : X-axis means prallel to B-field, Y-axis means perpendicular to B-field
function SFr_Global(Vx::Cube,Vz::Cube,Vy::Cube,bx::Cube,by::Cube,bz::Cube)
  Nx,Ny,Nz = size(Vx);
  R        = sqrt(s(div(Nx,2),div(Ny,2),div(Nz,2)));

  #get the mean field
  mbx= mean(bx);
  mby= mean(by);
  mbz= mean(bz);
    
  #get the structure function 
  SFVx  = SFC(Vx);
  SFVy  = SFC(Vy);
  SFVz  = SFC(Vz);
  #get the vector structure function
  SFV   = @. SFVx + SFVy + SFVz;
  #declaring the output
  Rx,Ry = RoundUpInt(R), RoundUpInt(R);
  Mask = zeros((Rx+1,Ry+1));
  SFVr = zeros((Rx+1,Ry+1));

  for k in 1:Nz, j in 1:Ny, i in 1:Nx
    # get the k vector
    idx = i-div(Nx,2);
    jdx = j-div(Ny,2);
    kdx = k-div(Nz,2);
    kk  = sqrt(s(idx,jdx,kdx));
    if kk>0
      #get the θ between b unit vector and k vector
      θ  = t(mbx,mby,mbz,idx,jdx,kdx)

      #get the 2D vector parallel and perpendicular to B-field
      kll = abs(kk*cos(θ));
      kpp = abs(kk*sin(θ));

      rpar,rperp       = getind(kll,kpp,Rx,Ry).+1; #prob2
      Mask[rpar,rperp] += 1; 
      SFVr[rpar,rperp] += SFV[i,j,k];
    end
  end
  SFVr./=Mask;
  SFVr
end

#provding 2D 2 point local structure point function for vector (iv,jv,kv)
#Note : X-axis means prallel to B-field, Y-axis means perpendicular to B-field
@inline function SFr_local_CPU(iv::Array{T,3},jv::Array{T,3},kv::Array{T,3},
                               ib::Array{T,3},jb::Array{T,3},kb::Array{T,3}; Nseed=600,r=100) where T
  #println("Excepted waiting time ~ 0.5s for r=100, Nseed=1 ")
  t_s = round(0.5*Nseed*(r/100)^3,digits=1);
  #println("You waiting time ~ ",t_s,"s for r =",r,", N=",Nseed);
  nx,ny,nz = size(iv);
  x0,y0,z0 = div(nx,2),div(nx,2), div(nx,2);
  x_seed = rand(r+1:nx-r,(Nseed));
  y_seed = rand(r+1:ny-r,(Nseed));
  z_seed = rand(r+1:nz-r,(Nseed));
  mb     = zeros(3);        
    
  # get the size of the structure function array
  R     = sqrt(3*(r+1)^2);
  Rx,Ry = round(Int,R,RoundUp),round(Int,R,RoundUp);

  # declaring the output 
  Mask = zeros((Rx+2,Ry+2));
  SFVr = zeros((Rx+2,Ry+2));

  p = Progress(Nseed,
           barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
           barlen=10, showspeed=true);
    
  @inbounds begin
    Threads.@threads for seed = 1:Nseed
    xs = x_seed[seed]::Int;
    ys = y_seed[seed]::Int;
    zs = z_seed[seed]::Int;
    iv0,jv0,kv0 = iv[xs,ys,zs]::T,jv[xs,ys,zs]::T,kv[xs,ys,zs]::T;
    ib0,jb0,kb0 = ib[xs,ys,zs]::T,jb[xs,ys,zs]::T,kb[xs,ys,zs]::T;
    for k = zs-r : zs+r
      kdx = k-zs;
      for j = ys-r : ys+r
        jdx = j-ys;
        @simd  for i = xs-r:xs+r
          idx = i-xs;
          kk  = sqrt(s(idx,jdx,kdx))::Float64;
          if kk > 0
            @muladd begin
            ibi, jbi, kbi = ib[i,j,k]::T,jb[i,j,k]::T,kb[i,j,k]::T;
            ivi, jvi, kvi = iv[i,j,k]::T,jv[i,j,k]::T,kv[i,j,k]::T;          
            #let the θ between b unit vector and k vector
            mb1 = (ib0+ibi)*0.5;
            mb2 = (jb0+jbi)*0.5;
            mb3 = (kb0+kbi)*0.5;
            θ  = t(mb1,mb2,mb3,idx,jdx,kdx);
            #get the 2D vector parallel and perpendicular to B-field
            kll = abs(kk*cos(θ));
            kpp = abs(kk*sin(θ));
            #get the 2D vector parallel and perpendicular to B-field
            rpar        = round(Int,kll);
            rperp       = round(Int,kpp);
            Mask[rpar+1,rperp+1] += 1; 
            SFVr[rpar+1,rperp+1] += s(iv0-ivi,jv0-jvi,kv0-kvi);
            end
          end
        end
      end
    end
    next!(p)
  end
  end
  SFVr./=Mask
  return SFVr;
end

function SFr_local_GPU(iv,jv,kv,ib,jb,kb; Nseed=1000)
  if length(CUDA.devices()) > 0
     nx,ny,nz = size(iv);
     x_pairs = CuArray((rand(1:nx,(Nseed))))
     y_pairs = CuArray((rand(1:ny,(Nseed))))
     z_pairs = CuArray((rand(1:nz,(Nseed))))
     # Define the output array
     R     = √((nx/2)^2+(ny/2)^2+(nz/2)^2);
     Rx,Ry = round(Int,R,RoundUp),round(Int,R,RoundUp);
     Mask = CUDA.zeros(Float32,(Rx+2,Ry+2));
     SFVr = CUDA.zeros(Float32,(Rx+2,Ry+2));
     
     threads = ( 32, 8, 1)
     blocks   = ( ceil(Int,size(iv,1)/threads[1]), ceil(Int,size(jv,2)/threads[2]), ceil(Int,size(kv,3)/threads[3]))  
     CUDA.@time begin
       @cuda blocks = blocks threads = threads local_SF_CUDA!(CuArray(iv),CuArray(jv),CuArray(kv),
                                                              CuArray(ib),CuArray(jb),CuArray(kb),
                                                              x_pairs,y_pairs,z_pairs,
                                                                Mask,SFVr,R)
     end
     return (Array(SFVr)./Array(Mask))
   else
     error("No GPU have been found!\n")
     return nothing
  end
end

#CUDA kenerl function
function local_SF_CUDA!(iv::CuArray{T,3},jv::CuArray{T,3},kv::CuArray{T,3},
                        ib::CuArray{T,3},jb::CuArray{T,3},kb::CuArray{T,3},
                        xps,yps,zps,
                        Mask,SFVr,R) where T
  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 
  if i < size(iv,1) && j < size(iv,2) && k < size(iv,3)
    @inbounds ib0, jb0, kb0 = ib[i,j,k],jb[i,j,k],kb[i,j,k];
    @inbounds iv0, jv0, kv0 = iv[i,j,k],jv[i,j,k],kv[i,j,k];
    for ri = 1:length(xps)
      @inbounds xp,yp,zp = xps[ri],yps[ri],zps[ri]
      idx, jdx, kdx = (xp - i), (yp - j), (zp - k)
      kk  = √(idx*idx + jdx*jdx + kdx*kdx);
      kk  = kk > R ? R - kk : kk
      if R > kk > 0
        ibi, jbi, kbi = ib[xp,yp,zp],jb[xp,yp,zp],kb[xp,yp,zp];
        ivi, jvi, kvi = iv[xp,yp,zp],jv[xp,yp,zp],kv[xp,yp,zp];    
        #let the θ between b unit vector and k vector
        mb1,mb2,mb3 = (ib0+ibi)*0.5,(jb0+jbi)*0.5,(kb0+kbi)*0.5
        t  = (mb1*idx + mb2*jdx + mb3*kdx)/sqrt(mb1^2+mb2^2+mb3^2)/sqrt(idx^2+jdx^2+kdx^2)
        t  = t > 1 ? 1 : t < -1 ? -1 : t 
        θ  = acos(t)
        #get the 2D vector parallel and perpendicular to B-field
        kll = abs(kk*cos(θ));
        kpp = abs(kk*sin(θ));
        #get the 2D vector parallel and perpendicular to B-field
        rpar        = round(Int,kll);
        rperp       = round(Int,kpp);
        @inbounds Mask[rpar+1,rperp+1] += 1; 
        @inbounds SFVr[rpar+1,rperp+1] += sqrt((iv0-ivi)^2 + (jv0-jvi)^2 + (kv0-kvi)^2);
      end
    end
  end
  return nothing
end

# Fitting Related function
# Contour function
function Getcontour(V::Mat,levels;cmap="Blue_r",Conmap="winter")
  imshow(V,cmap=cmap)
  Nx,Ny = size(V);
  A = contour(V,levels=levels,colors="black")
  axis([1,Ny,1,Nx])
  Conlevel = length(A.allsegs)
  x = zeros(Float64,Conlevel);
  y = zeros(Float64,Conlevel);
  for i = 1:Conlevel
    if length(A.allsegs[i])>0
      if (A.allsegs[i][1][1,1]!=0 && A.allsegs[i][end,end]!=0)
          x[i],y[i]  = NaN,NaN;
      end
      x[i],y[i] = A.allsegs[i][1][1,end],A.allsegs[i][1][end,1]
    else
    x[i],y[i]  = NaN,NaN;
    end
  end
  x[.~isnan.(y)],y[.~isnan.(y)]
end

function fitline(xx::Array,yy::Array,label)
  ind = findall((xx.>0).&(yy.>0));
  x = xx[ind];
  y = yy[ind];
  line(x,p)=p[1].+x.*p[2];
  p = [0,y[2]-y[1]];
  xxx=curve_fit(line,x,y,p).param;
  m,C = round(xxx[2],digits=2),round(10^(xxx[1]),digits=3)
  plot([10.0].^x,[10.0].^line(x,xxx),label="α = $m , A = $C, "*label);
  loglog([10.0].^x,[10.0].^y,"o")
  legend()
  return xxx
end

end