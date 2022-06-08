# ----------
# Time Integrator Function for MHDFlows
# ----------

function TimeIntegrator!(prob,t₀ :: Number,N₀ :: Int;
                                       usr_dt = 0.0,
                                        diags = [],
                                  loop_number = 100,
							     	     save = false,
							         save_loc = "",
							         filename = "",
							          dump_dt = 0)

  # Check if save function related parameter
  if (save)

  	if length(save_loc) == 0 || length(filename) == 0 || dump_dt == 0
  		error("Save Function Turned ON but save_loc/filename/dump_dt is not declared!\n");
  	end 
  end
  
  # Declare the vars update function and CFL time calclator
  updatevars! = ifelse(prob.flag.b, MHDSolver.MHDupdatevars!,
                                    HDSolver.HDupdatevars!);
  updateCFL!  = ifelse(prob.flag.b, CFL_MHD!, CFL_HD!);

  # Declare the iterator paramters
  Nᵢ = 0;
  t_next_save = prob.clock.t + dump_dt;
  
  # Check if user is declared a looping dt
  usr_declared_dt = usr_dt != 0.0 ? true : false 
  if (usr_declared_dt)
    prob.clock.dt = usr_dt;
  end

  while (N₀ >= Nᵢ) && (t₀ >= prob.clock.t)	
    if (!usr_declared_dt)
        #update the CFL condition;
      	updateCFL!(prob)
    end
  
  	#update the system; 
  	stepforward!(prob.sol, prob.clock, prob.timestepper, prob.eqn, 
                 prob.vars, prob.params, prob.grid);

    #update the diags
    increment!(diags)

    #Corret v and b if VP method is turned on
    if (prob.flag.vp == true)
      MHDSolver_VP.DivVCorrection!(prob);
      prob.flag.b == true ? MHDSolver_VP.DivBCorrection!(prob) : nothing;
    end

    #Dye Update
    prob.dye.dyeflag == true ? prob.dye.stepforward!(prob) : nothing;

  	#User defined function
  	for foo! ∈ prob.usr_func
  		foo!(prob);
  	end

  	#update the vars
	updatevars!(prob);

	Nᵢ += 1;
		
	#Save Section	
	if (save) && prob.clock.t >= t_next_save;
      KE_ = ProbDiagnostic(prob, Nᵢ; print_ = false);
      isnan(KE_) ? error("detected NaN! Quite the simulation right now.") : nothing;
      savefile(prob, file_number; file_path_and_name = file_path_and_name)
      t_next_save += dump_dt;
      file_number +=1;
    end

    if Nᵢ % loop_number == 0
    	KE_ = ProbDiagnostic(prob, Nᵢ; print_ = true);
        isnan(KE_) ? error("detected NaN! Quite the simulation right now.") : nothing;
    end

  end

end

function CFL_MHD!(prob;Coef = 0.25)
    #Solving the dt of CFL condition using dt = Coef*dx/v
    
	#Maxmium velocity 
    @. prob.vars.nonlin1 *=0;
    v2 = prob.vars.nonlin1;
    v2 += prob.vars.ux.^2 + prob.vars.uy.^2 + prob.vars.uz.^2;
    v2_max = maximum(v2);

	#Maxmium Alfvenic velocity 
    @. prob.vars.nonlin1 *=0;
    v2a = prob.vars.nonlin1;
    v2a += prob.vars.bx.^2 + prob.vars.by.^2 + prob.vars.bz.^2;
    v2a_max = maximum(v2a);
 
    vmax = sqrt(maximum([v2_max,v2a_max]));
    dx = prob.grid.Lx/prob.grid.nx;
    dy = prob.grid.Ly/prob.grid.ny;
    dz = prob.grid.Lz/prob.grid.nz;
    dl = minimum([dx,dy,dz]);
    dt =  Coef*dl/vmax;
    prob.clock.dt = dt;

end


function CFL_HD!(prob;Coef = 0.25)
    #Solving the dt of CFL condition using dt = Coef*dx/v
    
	#Maxmium velocity 
    @. prob.vars.nonlin1 *=0;
    v2 = prob.vars.nonlin1;
    v2 += prob.vars.ux.^2 + prob.vars.uy.^2 + prob.vars.uz.^2;
    vmax = sqrt(maximum(v2));

    dx = prob.grid.Lx/prob.grid.nx;
    dy = prob.grid.Ly/prob.grid.ny;
    dz = prob.grid.Lz/prob.grid.nz;
    dl = minimum([dx,dy,dz]);
    dt =  Coef*dl/vmax;
    prob.clock.dt = dt;

end

function ProbDiagnostic(prob,N; print_=false)
    dV = (2π/prob.grid.nx)^3;
    vx,vy,vz = prob.vars.ux,prob.vars.uy,prob.vars.uz;
    KE =  string(round(sum(vx.^2+vy.^2 + vz.^2)*dV,sigdigits=3));
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

function savefile(prob,ρₖ,file_number;file_path_and_name="")
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
