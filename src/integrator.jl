module integrator

include("Solver.jl")

using
  CUDA,
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

export MHDintegrator!

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


end