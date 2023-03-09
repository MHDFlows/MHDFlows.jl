# ----------
# Time Integrator Function for MHDFlows
# ----------

"""
    TimeIntegrator!(prob,t₀,N₀,
                 usr_dt = 0.0,
               CFL_Coef = 0.25,
           CFL_function = nothingfunction,
                  diags = [],
      dynamic_dashboard = true,
            loop_number = 100,
                   save = false,
               save_loc = "",
               filename = "",
            file_number = 0,
                dump_dt = 0)

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
"""
function TimeIntegrator!(prob,t₀ :: Number,N₀ :: Int;
                                       usr_dt = 0.0,
                                     CFL_Coef = 0.25,
                                 CFL_function = nothingfunction,
                                        diags = [],
                            dynamic_dashboard = true,
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
    vi = prob.flag.e ?  prob.params.η : maximum([prob.params.ν,prob.params.η])
    nv = prob.flag.e ? prob.params.nη : maximum([prob.params.nν,prob.params.nη])
  else
    vi = prob.params.ν
    nv = prob.params.nν
  end
  dx = prob.grid.Lx/prob.grid.nx
  dy = prob.grid.Ly/prob.grid.ny
  dz = prob.grid.Lz/prob.grid.nz
  dl = minimum([dx,dy,dz])
  t_diff = ifelse(nv >1, CFL_Coef*(dl)^(nv)/vi, CFL_Coef*dl^2/vi)

  # Declare the iterator paramters
  t_next_save = prob.clock.t + dump_dt
  prob.clock.step = 0
  
  # Check if user is declared a looping dt
  usr_declared_dt = usr_dt != 0.0 ? true : false 
  if (usr_declared_dt)
    prob.clock.dt = usr_dt
  end

  # Clean the divgence of b if VP method is turned on
  if (prob.flag.vp == true)
    VPSolver.DivVCorrection!(prob)
    prob.flag.b == true ? VPSolver.DivBCorrection!(prob) : nothing
  end

  # Print the wellcome message
  WellcomeMessage()

  # check if user enable the dynamical dashboard
  if dynamic_dashboard 
    prog = Progress(N₀; desc = "Simulation in rogress :", 
                        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
                        barlen=10, showspeed=true)
  else
    prog = nothing
  end

  # Actual Computation Start
  time = @elapsed begin
    while (N₀ >= prob.clock.step ) && (t₀ >= prob.clock.t)  

      if (!usr_declared_dt)
        #update the CFL condition;
        updateCFL!(prob, t_diff; Coef = CFL_Coef)
      end

      #update the system; 
      stepforward!(prob.sol, prob.clock, prob.timestepper, prob.eqn, 
                   prob.vars, prob.params, prob.grid)

      #update the diags
      increment!(diags)

      #Corret b if VP method is turned on
      if (prob.flag.vp == true)
        VPSolver.DivVCorrection!(prob)
        prob.flag.b == true ? VPSolver.DivBCorrection!(prob) : nothing
      end

      #Dye Update
      prob.dye.dyeflag == true ? prob.dye.stepforward!(prob) : nothing

      #Shear Update
      prob.flag.s == true ? Shear.Shearing_remapping!(prob) : nothing      

      #User defined function
      for foo! ∈ prob.usr_func
          foo!(prob)
      end
          
      #Save Section   
      if (save) && prob.clock.t >= t_next_save
        ProbDiagnostic(prob)
        savefile(prob, file_number; file_path_and_name = file_path_and_name)
        t_next_save += dump_dt
        file_number +=1
      end

      # Update the dashboard information to user
      dynamic_dashboard ? Dynamic_Dashboard(prob,prog, N₀,t₀) : 
                            Static_Dashbroad(prob,prob.clock.step% loop_number)
    end

  end

  Ntotal = prob.grid.nx*prob.grid.ny*prob.grid.nz;
  Total_Update_per_second = prob.clock.step* Ntotal/time;
  print("Total CPU/GPU time run = $(round(time,digits=3)) s," 
        *" zone update per second = $(round(Total_Update_per_second,digits=3)) \n")
  return nothing

end

function getCFL!(prob, t_diff; Coef = 0.3)
  #Solving the dt of CFL condition using dt = Coef*dx/v
  square_maximum(A) =  mapreduce(x->x*x,max,A)
  if prob.flag.e
    # Maxmium velocity, For EMHD, v = ∇×B
    ux,uy,uz = prob.vars.∇XBᵢ , prob.vars.∇XBⱼ, prob.vars.∇XBₖ;
    v2xmax = square_maximum(ux)
    v2ymax = square_maximum(uy)
    v2zmax = square_maximum(uz)
    vmax = sqrt(maximum((v2xmax,v2ymax,v2zmax)))
  else
    #Maxmium velocity 
    ux,uy,uz = prob.vars.ux, prob.vars.uy,prob.vars.uz
    v2xmax = square_maximum(ux)
    v2ymax = square_maximum(uy)
    v2zmax = square_maximum(uz)
    vmax = sqrt(maximum((v2xmax,v2ymax,v2zmax)))
  end

  if prob.flag.b
    #Maxmium Alfvenic velocity 
    bx,by,bz = prob.vars.bx, prob.vars.by,prob.vars.bz
    v2xmax = square_maximum(bx)
    v2ymax = square_maximum(by)
    v2zmax = square_maximum(bz)
    vamax = sqrt(maximum([v2xmax,v2ymax,v2zmax]))
    vmax = maximum([vmax,vamax])
  end
    
  if prob.flag.c 
    vmax = maximum([vmax,prob.params.cₛ])
  end

  dx = prob.grid.Lx/prob.grid.nx
  dy = prob.grid.Ly/prob.grid.ny
  dz = prob.grid.Lz/prob.grid.nz
  # EMHD have two non-linear term, which makes dt ∝ Δx² in stead of Δx
  dl =  prob.flag.e ? minimum([dx,dy,dz])^2 : minimum([dx,dy,dz])
  dt = minimum([Coef*dl/vmax,t_diff])
  prob.clock.dt = dt
end

function CFL_Init(CFL_function::Function,usr_dt::Number)
  if usr_dt > 0.0
    error("User define both CFL_function and usr_dt")
  elseif usr_dt == 0.0
    return CFL_function
  end
end

function Restart!(prob,file_path_and_name)
  f = h5open(file_path_and_name,"r")

  if !prob.flag.e
    ux = read(f,"i_velocity")
    uy = read(f,"j_velocity")
    uz = read(f,"k_velocity")
    
    #Update V Conponment
    Move_Data_to_Prob!(ux, prob.vars.ux, view(prob.sol,:, :, :, prob.params.ux_ind),prob.grid)
    Move_Data_to_Prob!(uy, prob.vars.uy, view(prob.sol,:, :, :, prob.params.uy_ind),prob.grid)
    Move_Data_to_Prob!(uz, prob.vars.uz, view(prob.sol,:, :, :, prob.params.uz_ind),prob.grid)
  end

  #Update B Conponment
  if prob.flag.b == true
    bx = read(f,"i_mag_field",)
    by = read(f,"j_mag_field",)
    bz = read(f,"k_mag_field",)

    Move_Data_to_Prob!(bx,prob.vars.bx, view(prob.sol,:, :, :, prob.params.bx_ind),prob.grid)
    Move_Data_to_Prob!(by,prob.vars.by, view(prob.sol,:, :, :, prob.params.by_ind),prob.grid)
    Move_Data_to_Prob!(bz,prob.vars.bz, view(prob.sol,:, :, :, prob.params.bz_ind),prob.grid)
  end

  #Update the density
  if (prob.flag.c == true)
    ρ = read(f,"gas_density",)
    Move_Data_to_Prob!(ρ, prob.vars.ρ, view(prob.sol,:, :, :, prob.params.ρ_ind),prob.grid)
  end

  #if prob.flag.vp == true
  #  χ = read(f,"chi");
  #  copyto!(prob.params.χ, deepcopy(χ));
  #end

  #Update Dye
  if prob.dye.dyeflag == true
    ρ = read(f,"dye_density")
    copyto!(prob.dye.ρ, ρ)
    ρh  = prob.dye.tmp.sol₀
    mul!(ρh, prob.grid.rfftplan, prob.dye.ρ)
  end

  # Update time 
  prob.clock.t = read(f,"time")
  close(f)

  return nothing
end

function savefile(prob,file_number;file_path_and_name="")
  space_0 = ""
  for i = 1:4-length(string(file_number));space_0*="0";end
  fw = h5open(file_path_and_name*"_t_"*space_0*string(file_number)*".h5","w")
  if !prob.flag.e
    write(fw, "i_velocity",  Array(prob.vars.ux))
    write(fw, "j_velocity",  Array(prob.vars.uy))
    write(fw, "k_velocity",  Array(prob.vars.uz))
  end
  if (prob.dye.dyeflag == true)
    write(fw, "dye_density",  Array(prob.dye.ρ))
  end
  if (prob.flag.b == true)
    write(fw, "i_mag_field", Array(prob.vars.bx))
    write(fw, "j_mag_field", Array(prob.vars.by))
    write(fw, "k_mag_field", Array(prob.vars.bz))
  end

  if (prob.flag.c == true)
    write(fw, "gas_density", Array(prob.vars.ρ))
  end

  #if (prob.flag.vp == true)
  #    write(fw, "chi", Array(prob.params.χ));
  #end

  write(fw, "time", prob.clock.t)
  close(fw)
  return nothing
end