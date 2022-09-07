# ----------
# Time Integrator Function for MHDFlows
# ----------



"""
Time Integrator for MHDFlows problem 
  Keyword arguments
=================
- `prob`: MHDFlows problem 
- `N₀` : total iteration before stopping the integration (T type: Int)
- `t₀` : total time before stopping the integration (T type: Float Number)
- `usr_dt` : user defined time intreval for integration (T type: Float Number)
- `CFL_Coef` : CFL Ceof. (T type : Float Number)
- `CFL_function` : user defined CFL function 
- `loop_number` : iteration count for displaying the diagnostic information (T type : Int)
- `save` : save option for saving the hdf5 file (T type: true/false)
$(TYPEDFIELDS)
"""
function TimeIntegrator!(prob,t₀ :: Number,N₀ :: Int;
                                       usr_dt = 0.0,
                                     CFL_Coef = 0.25,
                                 CFL_function = nothingfunction,
                                        diags = [],
                                  loop_number = 100,
                                         save = false,
                                     save_loc = "",
                                     filename = "",
                                  file_number = 0,
                                      dump_dt = 0)
    
  # Check if save function related parameter
  if (save)
    if length(save_loc) == 0 || length(filename) == 0 || dump_dt == 0
        error("Save Function Turned ON but save_loc/filename/dump_dt is not declared!\n");
    end 
    file_path_and_name = save_loc*filename;
    savefile(prob, file_number; file_path_and_name = file_path_and_name);
    file_number+=1;
  end
  
  if CFL_function == nothingfunction
    updateCFL!  = getCFL!;
  else
    updateCFL!  = CFL_Init(CFL_function,usr_dt);
  end

  # Declare the timescale for diffusion
  if prob.flag.b
    vi = maximum([prob.params.ν,prob.params.η]);
    nv = maximum([prob.params.nν,prob.params.nη]);
  else
    vi = prob.params.ν;
    nv = prob.params.nν
  end
  dx = prob.grid.Lx/prob.grid.nx;
  dy = prob.grid.Ly/prob.grid.ny;
  dz = prob.grid.Lz/prob.grid.nz;
  dl = minimum([dx,dy,dz]);
  t_diff = ifelse(nv >1, CFL_Coef*(dl)^(2*nv)/vi,CFL_Coef*dl^2/vi);

  # Declare the iterator paramters
  t_next_save = prob.clock.t + dump_dt;
  prob.clock.step = 0;
  
  # Check if user is declared a looping dt
  usr_declared_dt = usr_dt != 0.0 ? true : false 
  if (usr_declared_dt)
    prob.clock.dt = usr_dt;
  end

  #Corret v and b if VP method is turned on
  if (prob.flag.vp == true)
    #MHDSolver_VP.DivVCorrection!(prob);
    prob.flag.b == true ? MHDSolver_VP.DivBCorrection!(prob) : nothing;
  end

  time = @elapsed begin
    while (N₀ >= prob.clock.step ) && (t₀ >= prob.clock.t)  

      if (!usr_declared_dt)
        #update the CFL condition;
        updateCFL!(prob, t_diff; Coef = CFL_Coef);
      end

      #update the system; 
      stepforward!(prob.sol, prob.clock, prob.timestepper, prob.eqn, 
                   prob.vars, prob.params, prob.grid);

      #update the diags
      increment!(diags)

      #Corret b if VP method is turned on
      if (prob.flag.vp == true)
        prob.flag.b == true ? MHDSolver_VP.DivBCorrection!(prob) : nothing;
      end

      #Dye Update
      prob.dye.dyeflag == true ? prob.dye.stepforward!(prob) : nothing;

      #User defined function
      for foo! ∈ prob.usr_func
          foo!(prob);
      end
          
      #Save Section   
      if (save) && prob.clock.t >= t_next_save;
        KE_ = ProbDiagnostic(prob, prob.clock.step; print_ = false);
        isnan(KE_) ? error("detected NaN! Quit the simulation right now.") : nothing;
        savefile(prob, file_number; file_path_and_name = file_path_and_name)
        t_next_save += dump_dt;
        file_number +=1;
      end

      if prob.clock.step % loop_number == 0
          KE_ = ProbDiagnostic(prob, prob.clock.step; print_ = true);
          isnan(KE_) ? error("detected NaN! Quit the simulation right now.") : nothing;
      end
    end
  end

  Ntotal = prob.grid.nx*prob.grid.ny*prob.grid.nz;
  Total_Update_per_second = prob.clock.step* Ntotal/time;
  print("Total CPU/GPU time run = $(round(time,digits=3)) s," 
        *" zone update per second = $(round(Total_Update_per_second,digits=3)) \n");
  return nothing;
end

function getCFL!(prob, t_diff; Coef = 0.3);
  #Solving the dt of CFL condition using dt = Coef*dx/v
  ux,uy,uz = prob.vars.ux, prob.vars.uy,prob.vars.uz;

  #Maxmium velocity 
  v2xmax = maximum(ux.^2);
  v2ymax = maximum(uy.^2);
  v2zmax = maximum(uz.^2);
  vmax = sqrt(maximum([v2xmax,v2ymax,v2zmax]));
  
  if prob.flag.b
    #Maxmium Alfvenic velocity 
    bx,by,bz = prob.vars.bx, prob.vars.by,prob.vars.bz;
    v2xmax = maximum(bx.^2);
    v2ymax = maximum(by.^2);
    v2zmax = maximum(bz.^2);
    vamax = sqrt(maximum([v2xmax,v2ymax,v2zmax]));
    vmax = maximum([vmax,vamax]);
  end
    
  dx = prob.grid.Lx/prob.grid.nx;
  dy = prob.grid.Ly/prob.grid.ny;
  dz = prob.grid.Lz/prob.grid.nz;
  dl = minimum([dx,dy,dz]);
  dt = minimum([Coef*dl/vmax,t_diff]);
  prob.clock.dt = dt;
end

function CFL_Init(CFL_function::Function,usr_dt::Number)
  if usr_dt > 0.0
    error("User define both CFL_function and usr_dt");
  elseif usr_dt == 0.0
    return CFL_function
  end
end

function ProbDiagnostic(prob,N; print_=false)
  dx,dy,dz = diff(prob.grid.x)[1],diff(prob.grid.y)[1],diff(prob.grid.z)[1];
  dV = dx*dy*dz;
  vx,vy,vz = prob.vars.ux,prob.vars.uy,prob.vars.uz;
  if prob.flag.vp
      χ  = prob.params.χ;  
      KE =  string(round(sum(vx[χ.==0].^2+vy[χ.==0].^2 + vz[χ.==0].^2)*dV,sigdigits=3));
  else
      KE =  string(round(sum(vx.^2+vy.^2 + vz.^2)*dV,sigdigits=3));
  end
  
  tt =  string(round(prob.clock.t,sigdigits=3));
  nn = string(N);
  for i = 1:8-length(string(tt));tt= " "*tt;end
  for i = 1:8-length(string(KE));KE= " "*KE;end
  for i = 1:8-length(string(nn));nn= " "*nn;end

  if (prob.flag.b == true)
      bx,by,bz = prob.vars.bx,prob.vars.by,prob.vars.bz;
      ME =  string(round(sum(bx.^2+by.^2 + bz.^2)*dV,sigdigits=3));
      for i = 1:8-length(string(ME));ME= " "*ME;end
      print_ == true ? println("n = $nn, t = $tt, KE = $KE, ME= $ME") : nothing;   
  else
      print_ == true ? println("n = $nn, t = $tt, KE = $KE") : nothing;  
  end

    return parse(Float32,KE)
end

function Restart!(prob,file_path_and_name)
  f = h5open(file_path_and_name,"r");
  ux = read(f,"i_velocity");
  uy = read(f,"j_velocity");
  uz = read(f,"k_velocity");
  
  #Update V Conponment
  copyto!(prob.vars.ux, deepcopy(ux));
  copyto!(prob.vars.uy, deepcopy(uy));
  copyto!(prob.vars.uz, deepcopy(uz));
  uxh = @view prob.sol[:, :, :, prob.params.ux_ind];
  uyh = @view prob.sol[:, :, :, prob.params.uy_ind];
  uzh = @view prob.sol[:, :, :, prob.params.uz_ind];
  mul!(uxh, prob.grid.rfftplan, prob.vars.ux);   
  mul!(uyh, prob.grid.rfftplan, prob.vars.uy);
  mul!(uzh, prob.grid.rfftplan, prob.vars.uz);
  copyto!(prob.vars.uxh, deepcopy(uxh));
  copyto!(prob.vars.uyh, deepcopy(uyh));
  copyto!(prob.vars.uzh, deepcopy(uzh));

  #Update B Conponment
  if prob.flag.b == true
    bx = read(f,"i_mag_field",);
    by = read(f,"j_mag_field",);
    bz = read(f,"k_mag_field",);

    copyto!(prob.vars.bx, deepcopy(bx));
    copyto!(prob.vars.by, deepcopy(by));
    copyto!(prob.vars.bz, deepcopy(bz));
    bxh = @view prob.sol[:, :, :, prob.params.bx_ind];
    byh = @view prob.sol[:, :, :, prob.params.by_ind];
    bzh = @view prob.sol[:, :, :, prob.params.bz_ind];
    mul!(bxh, prob.grid.rfftplan, prob.vars.bx);   
    mul!(byh, prob.grid.rfftplan, prob.vars.by);
    mul!(bzh, prob.grid.rfftplan, prob.vars.bz);
    copyto!(prob.vars.bxh, deepcopy(bxh));
    copyto!(prob.vars.byh, deepcopy(byh));
    copyto!(prob.vars.bzh, deepcopy(bzh));  
  end

  #if prob.flag.vp == true
  #  χ = read(f,"chi");
  #  copyto!(prob.params.χ, deepcopy(χ));
  #end

  #Update Dye
  if prob.dye.dyeflag == true; 
    ρ = read(f,"dye_density"); 
    copyto!(prob.dye.ρ, ρ);
    ρh  = prob.dye.tmp.sol₀;
    mul!(ρh, prob.grid.rfftplan, prob.dye.ρ);
  end

  # Update time 
  prob.clock.t = read(f,"time");
  close(f)
end

function savefile(prob,file_number;file_path_and_name="")
  space_0 = ""
  for i = 1:4-length(string(file_number));space_0*="0";end
  fw = h5open(file_path_and_name*"_t_"*space_0*string(file_number)*".h5","w")
  write(fw, "i_velocity",  Array(prob.vars.ux));
  write(fw, "j_velocity",  Array(prob.vars.uy));
  write(fw, "k_velocity",  Array(prob.vars.uz));
  if (prob.dye.dyeflag == true)
      write(fw, "dye_density",  Array(prob.dye.ρ));
  end
  if (prob.flag.b == true)
      write(fw, "i_mag_field", Array(prob.vars.bx));
      write(fw, "j_mag_field", Array(prob.vars.by));
      write(fw, "k_mag_field", Array(prob.vars.bz));
  end

  #if (prob.flag.vp == true)
  #    write(fw, "chi", Array(prob.params.χ));
  #end

  write(fw, "time", prob.clock.t);
  close(fw) 
end

function MHDintegrator!(prob,t)
    dt = prob.clock.dt;
    NStep = Int(round(t/dt));
    for i =1:NStep
        stepforward!(prob);
        MHDupdatevars!(prob);
    end
end

function HDintegrator!(prob,t)
    dt = prob.clock.dt;
    NStep = Int(round(t/dt));
    for i =1:NStep
        stepforward!(prob);
        HDupdatevars!(prob);
    end
end
