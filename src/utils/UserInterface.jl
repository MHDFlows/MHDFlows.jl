# Function Provding UserInface & Dashbroad function

function WellcomeMessage()
  println("=============================================================================");
  println("|                                                                           |");
  println("| |\\    /| |     | | ‾ ‾ \\     | ‾ ‾ ‾ |        / ‾ ‾ \\  |        | /‾‾‾‾\\  |");
  println("| | \\  / | | _ _ | |      |    |       |       |       | |        | \\____   |");
  println("| |  \\/  | |     | |      |    | ‾ ‾ ‾ |       |       |  \\  /\\  /       \\  |");
  println("| |      | |     | | _ _ /     |       | _ _ _  \\ _ _ /    \\/  \\/   \\____/  |");
  println("|                                                                           |");
  println("=============================================================================");

end

function WellcomeMessage2()
  println("============================================================================");
  println("|                                                                          |");
  println("|  M     M  H     H  D D D     F F F F  L          O O    W     W    S S   |");
  println("|  M M M M  H     H  D    D    F        L        O     O  W  W  W  S       |");
  println("|  M  M  M  H H H H  D     D   F F F F  L        O     O  W  W  W    S S   |");
  println("|  M     M  H     H  D    D    F        L        O     O  W  W  W       S  |");
  println("|  M     M  H     H  D D D     F        L L L L    O O      W W     S S    |");
  println("|                                                                          |");
  println("============================================================================");

end

# Function of computing the KE and ME
∑²(iv,jv,kv) = mapreduce((x,y,z)->(x^2 + y^2 + z^2),+,iv,jv,kv)

# Static dashboard for print n,KE,ME
function Static_Dashbroad(prob, step_over_check_loop_number::Number);

  tt = string(round(prob.clock.t,sigdigits=3));
  nn = string(prob.clock.step);
  for i = 1:8-length(string(tt));tt= " "*tt;end
  for i = 1:8-length(string(nn));nn= " "*nn;end

  if step_over_check_loop_number == 0
    if (prob.flag.b == true)
      if prob.flag.e
        ME  = ProbDiagnostic(prob);
        ME_ = string(ME);
        for i = 1:8-length(string(ME_));ME_= " "*ME_;end
        println("           n = $nn, t = $tt, ME = $ME_")
      else
        KE, ME  = ProbDiagnostic(prob);
        KE_,ME_ = string(KE),string(ME);
        for i = 1:8-length(string(KE_));KE_= " "*KE_;end
        for i = 1:8-length(string(ME_));ME_= " "*ME_;end
        println("            n = $nn, t = $tt, KE = $KE_, ME = $(ME)");
      end
    else
      KE  = ProbDiagnostic(prob);
      KE_ = string(KE);
      for i = 1:8-length(string(KE_));KE_= " "*KE_;end
      println("           n = $nn, t = $tt, KE = $KE_")
    end
  end
  return nothing

end

#Diagnostics function for static dashboard
function ProbDiagnostic(prob)
  dx,dy,dz = diff(prob.grid.x)[1],diff(prob.grid.y)[1],diff(prob.grid.z)[1];
  dV = dx*dy*dz;
  if !prob.flag.e
    vx,vy,vz = prob.vars.ux,prob.vars.uy,prob.vars.uz;
    KE = round(∑²(vx,vy,vz)*dV,sigdigits=3);
    isnan(KE) ? error("detected NaN! Quit the simulation right now.") : nothing;
  end
  if (prob.flag.b == true)
    bx,by,bz = prob.vars.bx,prob.vars.by,prob.vars.bz;
    ME = round(∑²(bx,by,bz)*dV,sigdigits=3);
    if !prob.flag.e
      return KE, ME
    else
      isnan(ME) ? error("detected NaN! Quit the simulation right now.") : nothing;
      return ME
    end
  else
    return KE
  end

end

# function for updating dynamical dashboard
function Dynamic_Dashboard(prob,prog,N₀,t₀)
  generate_showvalues(iter, Stats) = () -> [(:Progress,iter), (:Statistics,stats)];
  n = prob.clock.step;
  t    = round( prob.clock.t,sigdigits=3);
  dt   = round(prob.clock.dt,sigdigits=3);
  iter = "iter/Nₒ = $n/$(N₀), t/t₀ = $t/$(t₀), dt = $(dt)"
  if prob.flag.b 
    if prob.flag.e
      ME  = ProbDiagnostic(prob);
      stats = "ME = $(ME)"
    else
      KE, ME  = ProbDiagnostic(prob);
      stats = "KE = $(KE), ME = $(ME)"
    end
  else
    KE = ProbDiagnostic(prob);
    stats = "KE = $(KE)"
  end
  ProgressMeter.next!(prog; showvalues = generate_showvalues(iter,stats));
  return nothing
end
